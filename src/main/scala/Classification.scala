import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 8. 29..
  */
object Classification {

  def gradientBoostTrees(sc: SparkContext) = {
    import org.apache.spark.mllib.tree.GradientBoostedTrees
    import org.apache.spark.mllib.tree.configuration.BoostingStrategy
    import org.apache.spark.mllib.util.MLUtils

    // Load and parse the data file.
    val dir = "/Users/chanjinpark/GitHub/spark/"
    val data = MLUtils.loadLibSVMFile(sc, dir + "data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    // Classification에서는 Prediction값과 실제 값이 다르면 에러
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification GBT model:\n" + model.toDebugString)


    val boostingStrategy1 = BoostingStrategy.defaultParams("Regression")
    boostingStrategy1.numIterations = 3 // Note: Use more iterations in practice.
    boostingStrategy1.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy1.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model1 = GradientBoostedTrees.train(trainingData, boostingStrategy1)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions1 = testData.map { point =>
      val prediction = model1.predict(point.features)
      (point.label, prediction)
    }

    // Regression에서는 제곱의 평균 값으로 에러 측정
    val testMSE1 = labelsAndPredictions1.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Test Mean Squared Error = " + testMSE1)
    println("Learned regression GBT model:\n" + model1.toDebugString)

  }

  def randomForrest(sc: SparkContext) = {
    import org.apache.spark.mllib.tree.RandomForest
    import org.apache.spark.mllib.util.MLUtils

    // Load and parse the data file.
    val dir = "/Users/chanjinpark/GitHub/spark/"
    val data = MLUtils.loadLibSVMFile(sc, dir + "data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    println("Learned classification forest model:\n" + model.toDebugString)

    // Save and load model
    // model.save(sc, "target/tmp/myRandomForestClassificationModel")
    // val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")

    val impurity1 = "variance"
    val model1 = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity1, maxDepth, maxBins)

    val labelsAndPredictions1 = testData.map { point =>
      val prediction = model1.predict(point.features)
      (point.label, prediction)
    }
    val testMSE1 = labelsAndPredictions1.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Test Mean Squared Error = " + testMSE1)
    println("Learned regression forest model:\n" + model1.toDebugString)

    // Save and load model
    //model1.save(sc, "target/tmp/myRandomForestRegressionModel")
    //val sameModel1 = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")

  }


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf(true).setMaster("local").setAppName("Item2Item")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)



  }
}
