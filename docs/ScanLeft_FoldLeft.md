#### scanLeft vs. foldLeft
scanLeft는 중간결과를 저장하고, List로 리턴함
```scala
    val abc = Array("A", "B", "C")
    abc.scanLeft("z")(_ + _)
    // op: z + A = zA      // same operations as foldLeft above...
    // op: zA + B = zAB
    // op: zAB + C = zABC
    // res: List[String] = List(z, zA, zAB, zABC) // maps intermediate results

    abc.foldLeft("z")(_ + _) // with start value "z"
    // op: z + A = zA      // initial extra operation
    // op: zA + B = zAB
    // op: zAB + C = zABC
    // res: String = zABC

    abc.reduceLeft(_ + _)
    // op: A + B = AB
    // op: AB + C = ABC    // accumulates value AB in *first* operator arg `res`
    // res: String = ABC
```
#### aggregate() RDD function
Compared to reduce() & fold(), the aggregate() function has the advantage, it can return different Type vis-a-vis the RDD Element Type, ie Input Element type

    ```scala
    def aggregate[U](zeroValue: U)(seqOp: (U, T) ⇒ U, combOp: (U, U) ⇒ U)(implicit arg0: ClassTag[U]): U
    ```

* Aggregate the elements of each partition, and then the results for all the partitions, using given combine functions and a neutral "zero value".
* initial value to be used for both seqOp and combOp. zero 값은 파티션 마다 Merge할 때 마다 더해짐
* seqOp is used in every partition and combOp is used to combine results from every partition
* This function can return a different result type, U, than the type of this RDD, T.
* Thus, we need one operation for merging a T into an U and one operation for merging two U's, as in scala.TraversableOnce.
* Both of these functions are allowed to modify and return their first argument instead of creating a new U to avoid memory allocation.

    ```scala
    val inputrdd = sc.parallelize(List( ("maths", 21),("english", 22),("science", 31) ), 3)
    // seqOp for merging T into a U, ie (String, Int) in  into Int
    //  acc   :  Reprsents the accumulated result. value :  Represents the element in 'inputrdd' In our case this of type (String, Int)
    // combOp for mergining two U's  (ie 2 Int)
    val result = inputrdd.aggregate(3)( (acc, value) => (acc + value._2), (acc1, acc2) => (acc1 + acc2))
    //result: Int = 86
     ```