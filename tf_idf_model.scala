import org.apache.spark.mllib.linalg.{SparseVector => SV}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import breeze.linalg._

// run preprocessing.scala

val dim = math.pow(2,18).toInt
val hashingTF = new HashingTF(dim)
val tf = hashingTF.transform(tokens)
tf.cache

val v = tf.first.asInstanceOf[SV]
println(v.size)
println(v.values.size)
println(v.values.take(10).toSeq)
println(v.indices.take(10).toSeq)

val idf = new IDF().fit(tf)
val tfidf = idf.transform(tf)
val v2 = tfidf.first.asInstanceOf[SV]
println(v2.values.size)
println(v2.values.take(10).toSeq)
println(v2.indices.take(10).toSeq)

val minMaxVals = tfidf.map { v => val sv = v.asInstanceOf[SV]
(sv.values.min,sv.values.max)
}
val globalMinMax = minMaxVals.reduce { case ((min1, max1),(min2,max2)) => 
(math.min(min1,min2),math.max(max1,max2))
}
println(globalMinMax)

// check out TF-IDF values of some words - common words vs uncommon words
val common = sc.parallelize(Seq(Seq("you","do","we")))
val tfCommon = hashingTF.transform(common)
val tfidfCommon = idf.transform(tfCommon)
val commonVector = tfidfCommon.first.asInstanceOf[SV]
println(commonVector.values.toSeq)

val uncommon = sc.parallelize(Seq(Seq("telescope","legislation","investment")))
val tfUncommon = hashingTF.transform(uncommon)
val tfidfUncommon = idf.transform(tfUncommon)
val uncommonVector = tfidfUncommon.first.asInstanceOf[SV]
println(uncommonVector.values.toSeq)

// cosine similarity based on TF-IDF between two Documents
val hockeyText = rdd.filter { case (file,text) => file.contains("hockey")}
val hockeyTF = hockeyText.mapValues(doc => hashingTF.transform(tokenize(doc)))
val hockeyTfIdf = idf.transform(hockeyTF.map(_._2))

// two random documents on hockey and their cosine similairty
val hockey1 = hockeyTfIdf.sample(true,0.1,42).first.asInstanceOf[SV]
val breeze1 = new SparseVector(hockey1.indices,hockey1.values, hockey1.size)
val hockey2 = hockeyTfIdf.sample(true,0.1,43).first.asInstanceOf[SV]
val breeze2 = new SparseVector(hockey2.indices,hockey2.values,hockey2.size)
val cosineSim = breeze1.dot(breeze2) / (norm(breeze1)*norm(breeze2))
println(cosineSim)

val graphicsText = rdd.filter { case (file,text) => file.contains("comp.graphics")}
val graphicsTF = graphicsText.mapValues(doc => hashingTF.transform(tokenize(doc)))
val graphicsTfIdf = idf.transform(graphicsTF.map(_._2))
val graphics = graphicsTfIdf.sample(true, 0.1,42).first.asInstanceOf[SV]
val breezeGraphics = new SparseVector(graphics.indices,graphics.values,graphics.size)
val cosineSim2 = breeze1.dot(breezeGraphics) / (norm(breeze1) * norm(breezeGraphics))
println(cosineSim2)
