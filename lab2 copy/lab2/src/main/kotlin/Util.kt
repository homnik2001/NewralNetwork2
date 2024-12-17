package org.example

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.decodeFromStream
import java.io.File
import java.net.URL
import kotlin.math.pow

@OptIn(ExperimentalSerializationApi::class)
fun readWords(): Map<String, List<String>> {
    val inputDir = "inputdata/classes"
    val resourceDir: URL? = Thread.currentThread().contextClassLoader.getResource(inputDir)

    if (resourceDir == null) {
        println("Directory not found: $inputDir")
        return mutableMapOf()
    }
    val filesMap = mutableMapOf<String, List<String>>()

    File(resourceDir.toURI()).listFiles()?.forEach { file ->
        if (file.isFile) {
            val wordsArray = Json.decodeFromStream<List<String>>(file.inputStream())
            filesMap[file.name.replace(".json", "")] = wordsArray
        }
    }
    return filesMap
//    filesMap.forEach { k, v ->
//        println("##$k##")
//        v.forEach(System.out::println)
//    }
    //println(filesMap)
}

fun readNickNames(): List<String> {
    val jsonString = File("/Users/nikitakhomenko/Desktop/уеба/Neural Network/lab2/lab2/src/main/resources/inputdata/screen_names.json").readText()  // Считываем содержимое файла
    return Json.decodeFromString(jsonString)   // Декодируем в List<String>
}

fun splitStringIntoSubstringsByLength(input: String, minLength: Int): Map<Int, Array<String>> {
    val substringsByLength = mutableMapOf<Int, MutableSet<String>>()
    val length = input.length

    for (start in 0 until length) {
        for (end in start + minLength..length) {
            if (end <= length) {
                val substring = input.substring(start, end)
                val substringLength = substring.length

                if (!substringsByLength.containsKey(substringLength)) {
                    substringsByLength[substringLength] = mutableSetOf()
                }

                substringsByLength[substringLength]?.add(substring)
            }
        }
    }

    return substringsByLength.mapValues { it.value.toTypedArray() }
}

fun splitWordsToSubwords(words: List<String>): Map<Int, MutableMap<String, MutableList<String>>> {
    val ass = mutableMapOf<Int, MutableMap<String, MutableList<String>>>()
    words.forEach { word ->
        val splitResult = splitStringIntoSubstringsByLength(word, 2)
        //<sizeOfSubword, Map<SubWord,List<parentWordsOfSubWord> >>

        splitResult.forEach{ (wordSize, v) ->
            if(!ass.containsKey(wordSize)) {
                ass[wordSize] = mutableMapOf()
            }
            v.forEach{subWord ->
                if(ass[wordSize]?.containsKey(subWord) == false) {
                    ass[wordSize]?.put(subWord, mutableListOf())
                }
                ass[wordSize]?.get(subWord)?.add(word)
            }
        }
    }
    return ass;
}


fun calculateCount(data: Map<Int, MutableMap<String, MutableList<String>>>, word: String, classType: String): BestWordMatches {
    var classCounter = 0.0
    var maxSubWord: List<String> = listOf()
    var prevSize = 0
    var prevStr = ""
    data.forEach { (k,v) ->
        val multipleFactor = 1 + k.toDouble().pow(2.0)/16
        if(k >= 3)  {
            v.forEach { (k1, v1) ->
                if (word.contains(k1)) {
                    classCounter += (multipleFactor * (1 + v1.size.toDouble().pow(2.0) / 6))
                    if (prevSize < k) {
                        maxSubWord = v1
                        prevSize = k
                        prevStr = k1
                    }
                }
            }
        }
    }
    return BestWordMatches(classType, maxSubWord, classCounter, prevStr)
}

//fun splitWordToSubwords(word: String): List<String> {
//
//}


fun calculateCounts(data: Map<String, Map<Int, MutableMap<String, MutableList<String>>>>, words: List<String>): Map<String, MutableList<BestWordMatches>> {
    val mapS = mutableMapOf<String, MutableList<BestWordMatches>>()
    data.forEach { (k,v) ->
        words.forEach { word ->
            if(mapS.containsKey(word)) {
                mapS[word]?.add(calculateCount(v, word, k))
            } else {
                mapS[word] = mutableListOf(calculateCount(v, word, k))
            }
        }
    }
    return mapS
}
@Serializable
data class BestWordMatches(
    val classType: String,
    val maxSubWord: List<String>,
    val classCounter: Double,
    val matchesSubword: String
)
fun readResult():Map<String, MutableList<BestWordMatches>> {
    // Чтение из файла и десериализация
    val jsonData = File("/Users/nikitakhomenko/Desktop/уеба/Neural Network/lab2/lab2/src/main/resources/inputdata/resultFile.json").readText()
    return Json.decodeFromString(jsonData)

}
fun writeResult(bestWordMatchesList :Map<String, MutableList<BestWordMatches>>) {
    // Сериализация объекта в JSON и запись в файл
    val jsonString = Json.encodeToString(bestWordMatchesList)
    File("/Users/nikitakhomenko/Desktop/уеба/Neural Network/lab2/lab2/src/main/resources/inputdata/resultFile.json").writeText(jsonString)
}