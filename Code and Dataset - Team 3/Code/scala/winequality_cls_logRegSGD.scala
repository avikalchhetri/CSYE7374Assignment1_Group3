import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, L1Updater}
object SimpleApp {
  def main(args: Array[String]) {
// Load and parse the data
val data = sc.textFile("C:\\Users\\anirudhbedre\\Desktop\\Wine\\wine_quality_pyspark_classification.csv")
val parsedData = data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(11).toDouble, Vectors.dense(parts.slice(0,11).map(_.toDouble)))}.cache()


val splits = parsedData.randomSplit(Array(0.7, 0.3))
val training = splits(0).cache()
val test = splits(1).cache()
val numTraining = training.count()
val numTest = test.count()

val alg =new LogisticRegressionWithSGD()
alg.optimizer.setNumIterations(100)
alg.optimizer.setStepSize(0.000469)
val model=alg.run(training)  


//  Calculate training accuracy
val training_prediction = model.predict(training.map(_.features))
val train_valuesAndPreds = training_prediction.zip(training.map(_.label)) 
val metrics = new MulticlassMetrics(train_valuesAndPreds)
val precision = metrics.precision
println(" Training Accuracy = " + precision)

val test_prediction = model.predict(test.map(_.features))
val test_valuesAndPreds = test_prediction.zip(test.map(_.label)) 
val metrics = new MulticlassMetrics(test_valuesAndPreds)
val precision = metrics.precision
println(" Training Accuracy = " + precision)

}
}
  
  
  