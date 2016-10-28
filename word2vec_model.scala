import org.apache.spark.mllib.feature.Word2Vec

val word2vec = new Word2Vec()
word2vec.setSeed(42)
val word2vecModel = word2vec.fit(tokens)

word2vecModel.findSynonyms("hockey",20).foreach(println)

word2vecModel.findSynonyms("legislation",20).foreach(println)