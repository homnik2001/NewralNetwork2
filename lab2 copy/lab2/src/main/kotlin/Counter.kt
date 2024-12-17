package org.example

import org.example.dto.Order

class Counter {
    fun xxx() {
        val s = splitStringIntoSubstringsByLength("w", 2)
    }



    fun count(order: Order, subWords: Map<Int,Array<String>>): Int {
        var counter = 0
        var topWord: String?
        var topCount: Int?
        order.map.forEach{ (k, v) ->
            subWords.getValue(k).forEach { subWord ->
                if(v.contains(subWord)) {
                    counter+=k
                    topWord = subWord
                }
            }
        }
        return 0
    }
}