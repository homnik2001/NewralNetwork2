package org.example.dto

data class Order(
    val className: String,
    val xzc : Set<String>,
    val currentCount: Int,
    val map : Map<Int, Set<String>>
)
data class ZXC(
    val word: String,
    val map : Map<Int, Set<String>>
)
