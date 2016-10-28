import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// after run preprocessing.scala and tf_idf_model.scala

val newsgroupsMap = newsgroups.distinct.collect().zipWithIndex.toMap
val zipped = newsgroups.zip(tfidf)
val train = zipped.map { case (topic, vector) => LabeledPoint(newsgroupsMap(topic), vector)}
train.cache

val model = NaiveBayes.train(train, lambda = 0.1)

val testPath = "/Users/shirotsuchiya/Documents/spark_nlp/20news-bydate-test/*"
val testRDD = sc.wholeTextFiles(testPath)
val testLabels = testRDD.map { case (file, text) => 
    val topic = file.split("/").takeRight(2).head
    newsgroupsMap(topic)
}

val testTf = testRDD.map { case (file, text) => hashingTF.transform(tokenize(text))}
val testTfIdf = idf.transform(testTf)
val zippedTest = testLabels.zip(testTfIdf)
val test = zippedTest.map{ case (topic,vector) => LabeledPoint(topic,vector)}

val predictionAndLabel = test.map(p => (model.predict(p.features),p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
val metrics = new MulticlassMetrics(predictionAndLabel)
println(accuracy)
println(metrics.weightedFMeasure)