val path = "/Users/shirotsuchiya/Documents/spark_nlp/20news-bydate-train/*"
val rdd = sc.wholeTextFiles(path,4)
val text = rdd.map { case (file, text) => text}
println(text.count)

val newsgroups = rdd.map { case(file, text) => file.split("/").takeRight(2).head}
val countByGroup = newsgroups.map(n => (n,1)).reduceByKey(_+_).collect.sortBy(-_._2).mkString("\n")
println(countByGroup)

val text = rdd.map { case (file,text) => text}
val whiteSpaceSplit = text.flatMap(t => t.split(" ").map(_.toLowerCase))
println(whiteSpaceSplit.distinct.count)

// see a random sample of Documents
println(whiteSpaceSplit.sample(true,0.3,42).take(100).mkString(","))

// improve tokenization by applying regex
val nonWordSplit = text.flatMap(t => t.split("""\W+""").map(_.toLowerCase))
println(nonWordSplit.distinct.count)

// see the same sample 
println(whiteSpaceSplit.sample(true,0.3,42).take(100).mkString(","))

// remove numbers
val regex = """[^0-9]*""".r
val filterNumbers = nonWordSplit.filter(token => regex.pattern.matcher(token).matches)
println(filterNumbers.distinct.count)

// see the same sample to check the results
println(whiteSpaceSplit.sample(true,0.3,42).take(100).mkString(","))

// check top 20 most frequently occuring words
val tokenCounts = filterNumbers.map(t => (t,1)).reduceByKey(_+_)
val oreringDesc = Ordering.by[(String, Int), Int](_._2)
println(tokenCounts.top(20)(oreringDesc).mkString("\n"))

val stopwords = Set("the","a","an","of","or","in","for","by","on","but","is","not","with","as","was","if","they","are","this","and","it","have","from","at","my","be","that","to")

val tokenCountsFilteredStopwords = tokenCounts.filter {
	case (k,v) => !stopwords.contains(k)
}
println(tokenCountsFilteredStopwords.top(20)(oreringDesc).mkString("\n"))

// remove one letter word
val tokenCountsFilteredSize = tokenCountsFilteredStopwords.filter { case (k,v) => k.size >= 2}
println(tokenCountsFilteredSize.top(20)(oreringDesc).mkString("\n"))

// the least frequently occuring words
val oreringAsc = Ordering.by[(String,Int),Int](-_._2)
println(tokenCountsFilteredSize.top(20)(oreringAsc).mkString("\n"))

// remove words that appear only once
val rareTokens = tokenCounts.filter{ case (k,v) => v < 2}.map { case(k,v) => k}.collect.toSet
val tokenCountsFilteredAll = tokenCountsFilteredSize.filter { case(k,v) => !rareTokens.contains(k)}
println(tokenCountsFilteredAll.top(20)(oreringAsc).mkString("\n"))

println(tokenCountsFilteredAll.count)

// apply all the filters above in one function
def tokenize(line: String): Seq[String] = {
    line.split("""\W+""")
        .map(_.toLowerCase)
        .filter(token => 
            regex.pattern.matcher(token).matches)
        .filterNot(token => stopwords.contains(token))
        .filterNot(token => rareTokens.contains(token))
        .filter(token => token.size >= 2)
        .toSeq
}

// the same results can be obtained by the following:
println(text.flatMap(doc => tokenize(doc)).distinct.count)

// tokenize each Documents
val tokens = text.map(doc => tokenize(doc))
println(tokens.first.slice(0,20))