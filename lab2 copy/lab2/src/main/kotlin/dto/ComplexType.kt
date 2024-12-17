import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.io.File

@Serializable
data class ComplexType(
    val data: Map<String, Map<Int, MutableMap<String, MutableList<String>>>>
)

// Функция записи данных в файл
fun writeToFile(
    fileName: String,
    data: Map<String, Map<Int, MutableMap<String, MutableList<String>>>>
) {
    val json = Json { prettyPrint = true }
    val jsonString = json.encodeToString(ComplexType(data))
    File(fileName).writeText(jsonString)
}

// Функция чтения данных из файла
fun readFromFile(fileName: String): Map<String, Map<Int, MutableMap<String, MutableList<String>>>> {
    val json = Json { ignoreUnknownKeys = true }
    val jsonString = File(fileName).readText()
    val complexType = json.decodeFromString<ComplexType>(jsonString)
    return complexType.data
}
