### Spark Implementation of Gradient Boosting
boosting의 각 반복별로 에러와 손실을 계산함
* Labeled Point RDD와 loss 평가 메트릭을 입력으로 받음. 평가한 결과를 리턴

##### GradientBoostedTreesModel 클래스의 evaluateEachIteration 메소드 구현

```scala
 /**
   * Method to compute error or loss for every iteration of gradient boosting.
   * @param data RDD of [[org.apache.spark.mllib.regression.LabeledPoint]]
   * @param loss evaluation metric.
   * @return an array with index i having the losses or errors for the ensemble
   *         containing the first i+1 trees
   */
  @Since("1.4.0")
  def evaluateEachIteration(
      data: RDD[LabeledPoint],
      loss: Loss): Array[Double] = {

    val sc = data.sparkContext
    val remappedData = algo match {
      case Classification => data.map(x => new LabeledPoint((x.label * 2) - 1, x.features))
      case _ => data
    }

    val broadcastTrees = sc.broadcast(trees)
    val localTreeWeights = treeWeights
    val treesIndices = trees.indices

    val dataCount = remappedData.count()
    val evaluation = remappedData.map { point =>
      treesIndices
        .map(idx => broadcastTrees.value(idx).predict(point.features) * localTreeWeights(idx))
        .scanLeft(0.0)(_ + _).drop(1)
        .map(prediction => loss.computeError(prediction, point.label))
    }
    .aggregate(treesIndices.map(_ => 0.0))(
      (aggregated, row) => treesIndices.map(idx => aggregated(idx) + row(idx)),
      (a, b) => treesIndices.map(idx => a(idx) + b(idx)))
    .map(_ / dataCount)

    broadcastTrees.destroy()
    evaluation.toArray
  }
```

##### GradientBoostedTreesModel 클래스의 boost 메소드 구현

```scala
/**
   * Internal method for performing regression using trees as base learners.
   * @param input training dataset
   * @param validationInput validation dataset, ignored if validate is set to false.
   * @param boostingStrategy boosting parameters
   * @param validate whether or not to use the validation dataset.
   * @param seed Random seed.
   * @return tuple of ensemble models and weights:
   *         (array of decision tree models, array of model weights)
   */
  def boost(
      input: RDD[LabeledPoint],
      validationInput: RDD[LabeledPoint],
      boostingStrategy: OldBoostingStrategy,
      validate: Boolean,
      seed: Long): (Array[DecisionTreeRegressionModel], Array[Double]) = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters
    val numIterations = boostingStrategy.numIterations
    val baseLearners = new Array[DecisionTreeRegressionModel](numIterations)
    val baseLearnerWeights = new Array[Double](numIterations)
    val loss = boostingStrategy.loss
    val learningRate = boostingStrategy.learningRate
    // Prepare strategy for individual trees, which use regression with variance impurity.
    val treeStrategy = boostingStrategy.treeStrategy.copy
    val validationTol = boostingStrategy.validationTol
    treeStrategy.algo = OldAlgo.Regression
    treeStrategy.impurity = OldVariance
    treeStrategy.assertValid()

    // Cache input
    val persistedInput = if (input.getStorageLevel == StorageLevel.NONE) {
      input.persist(StorageLevel.MEMORY_AND_DISK)
      true
    } else {
      false
    }

    // Prepare periodic checkpointers
    val predErrorCheckpointer = new PeriodicRDDCheckpointer[(Double, Double)](
      treeStrategy.getCheckpointInterval, input.sparkContext)
    val validatePredErrorCheckpointer = new PeriodicRDDCheckpointer[(Double, Double)](
      treeStrategy.getCheckpointInterval, input.sparkContext)

    timer.stop("init")

    logDebug("##########")
    logDebug("Building tree 0")
    logDebug("##########")

    // Initialize tree
    timer.start("building tree 0")
    val firstTree = new DecisionTreeRegressor().setSeed(seed)
    val firstTreeModel = firstTree.train(input, treeStrategy)
    val firstTreeWeight = 1.0
    baseLearners(0) = firstTreeModel
    baseLearnerWeights(0) = firstTreeWeight

    var predError: RDD[(Double, Double)] =
      computeInitialPredictionAndError(input, firstTreeWeight, firstTreeModel, loss)
    predErrorCheckpointer.update(predError)
    logDebug("error of gbt = " + predError.values.mean())

    // Note: A model of type regression is used since we require raw prediction
    timer.stop("building tree 0")

    var validatePredError: RDD[(Double, Double)] =
      computeInitialPredictionAndError(validationInput, firstTreeWeight, firstTreeModel, loss)
    if (validate) validatePredErrorCheckpointer.update(validatePredError)
    var bestValidateError = if (validate) validatePredError.values.mean() else 0.0
    var bestM = 1

    var m = 1
    var doneLearning = false
    while (m < numIterations && !doneLearning) {
      // Update data with pseudo-residuals
      val data = predError.zip(input).map { case ((pred, _), point) =>
        LabeledPoint(-loss.gradient(pred, point.label), point.features)
      }

      timer.start(s"building tree $m")
      logDebug("###################################################")
      logDebug("Gradient boosting tree iteration " + m)
      logDebug("###################################################")
      val dt = new DecisionTreeRegressor().setSeed(seed + m)
      val model = dt.train(data, treeStrategy)
      timer.stop(s"building tree $m")
      // Update partial model
      baseLearners(m) = model
      // Note: The setting of baseLearnerWeights is incorrect for losses other than SquaredError.
      //       Technically, the weight should be optimized for the particular loss.
      //       However, the behavior should be reasonable, though not optimal.
      baseLearnerWeights(m) = learningRate

      predError = updatePredictionError(
        input, predError, baseLearnerWeights(m), baseLearners(m), loss)
      predErrorCheckpointer.update(predError)
      logDebug("error of gbt = " + predError.values.mean())

      if (validate) {
        // Stop training early if
        // 1. Reduction in error is less than the validationTol or
        // 2. If the error increases, that is if the model is overfit.
        // We want the model returned corresponding to the best validation error.

        validatePredError = updatePredictionError(
          validationInput, validatePredError, baseLearnerWeights(m), baseLearners(m), loss)
        validatePredErrorCheckpointer.update(validatePredError)
        val currentValidateError = validatePredError.values.mean()
        if (bestValidateError - currentValidateError < validationTol * Math.max(
          currentValidateError, 0.01)) {
          doneLearning = true
        } else if (currentValidateError < bestValidateError) {
          bestValidateError = currentValidateError
          bestM = m + 1
        }
      }
      m += 1
    }

    timer.stop("total")

    logInfo("Internal timing for DecisionTree:")
    logInfo(s"$timer")

    predErrorCheckpointer.deleteAllCheckpoints()
    validatePredErrorCheckpointer.deleteAllCheckpoints()
    if (persistedInput) input.unpersist()

    if (validate) {
      (baseLearners.slice(0, bestM), baseLearnerWeights.slice(0, bestM))
    } else {
      (baseLearners, baseLearnerWeights)
    }
  }
```