package org.example

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import readFromFile
import writeToFile
import java.io.File

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
fun main() {
    val inputString = "amamacriminal"

    readAndSaveDataToJson()
    val data =  readDataFromJson()
    val nicknames = readNickNames()
    val res =  calculateCounts(data, nicknames)
//    //writeResult(res)

        //    val res = readResult()
    val resultMap = res.mapValues { entry -> entry.value.maxByOrNull { it.classCounter } }


    // Получаем Map с ключами и соответствующими classType
    val classTypeMap: Map<String, String> = resultMap.mapValues { it.value?.classType.toString() }

    // Группируем ключи по значениям classType
    val typeToNickNamesListMap: Map<String, List<String>> = classTypeMap
        .entries
        .groupBy({ it.value }, { it.key })

    val jsonString = Json.encodeToString(classTypeMap)
    File("/Users/nikitakhomenko/Desktop/уеба/Neural Network/lab2/lab2/src/main/resources/inputdata/classToNickName.json").writeText(jsonString)

    val classTypeCount: Map<String, List<BestWordMatches?>> = resultMap.values
        .groupBy { it?.classType.toString() }                // Подсчитываем количество для каждой группы

    // Выводим результаты
    classTypeCount.forEach { (classType, count) ->
        println("$classType -> $count")
    }
    println("$inputString")
}

fun readAndSaveDataToJson() {
    val wordsByClasses = readWords()
    val wordsByClassesMap = mutableMapOf<String, Map<Int, MutableMap<String, MutableList<String>>>>()
    wordsByClasses.forEach { (k, v) ->
        wordsByClassesMap[k] = splitWordsToSubwords(v)
    }
    writeToFile("/Users/nikitakhomenko/Desktop/уеба/Neural Network/lab2/lab2/src/main/resources/megastructure.json", wordsByClassesMap)
}

fun readDataFromJson(): Map<String, Map<Int, MutableMap<String, MutableList<String>>>> {
    return readFromFile("/Users/nikitakhomenko/Desktop/уеба/Neural Network/lab2/lab2/src/main/resources/megastructure.json")
}

