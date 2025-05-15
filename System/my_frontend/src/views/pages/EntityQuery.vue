<template>
  <div class="bg-white p-6 rounded-lg shadow-md">
    <h2 class="text-xl font-semibold text-gray-800 mb-4">实体信息检索</h2>

    <!-- 输入框 -->
    <div class="flex gap-3 mb-6">
      <input
        v-model="query"
        class="flex-1 px-4 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
        placeholder="请输入你要查询的实体..."
      />
      <button
        @click="search"
        class="px-5 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        检索
      </button>
    </div>

    <!-- 图谱展示 -->
    <div v-if="graphData" class="my-6">
      <h3 class="text-lg font-bold mb-2 text-gray-700">相关图谱展示</h3>
      <div id="entity-graph" style="height: 600px; border: 1px solid #ccc;"></div>
    </div>

    <!-- 词云展示 -->
    <div class="bg-gray-50 p-4 rounded border">
      <h3 class="text-lg font-bold mb-2 text-gray-700">实体词云展示</h3>
      <div class="flex flex-wrap gap-4">
        <span
          v-for="word in politicalKeywords"
          :key="word"
          class="text-gray-600 hover:text-black cursor-pointer"
          :style="{ fontSize: getRandomFontSize() + 'px' }"
          @click="selectWord(word)"
        >
          {{ word }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { Network } from 'vis-network/standalone'

const query = ref('')
const graphData = ref(null)

// 政治热点关键词列表
const politicalKeywords = [
  'Tariff', 'Trade War', 'China', 'U.S', 'EU',
  'Global Economy', 'Foreign Policy', 'WTO', 'Trump', 'Biden', 'Vance'
]

// 点击词云项时的处理函数
function selectWord(word) {
  query.value = word
  search()
}

// 搜索函数
async function search() {
  if (!query.value) return
  try {
    const res = await axios.get(`http://localhost:8000/entity-graph?name=${encodeURIComponent(query.value)}`)
    graphData.value = res.data

    // 渲染图谱
    const container = document.getElementById('entity-graph')
    new Network(container, res.data, {
      nodes: {
        shape: 'dot',
        size: 20,
        font: { size: 14, color: '#000' },
        borderWidth: 2
      },
      edges: {
        arrows: 'to',
        font: { align: 'middle' },
        color: '#888'
      },
      physics: { stabilization: false }
    })
  } catch (err) {
    console.error('❌ 实体检索失败：', err)
  }
}

// 随机生成字体大小
function getRandomFontSize() {
  return Math.floor(Math.random() * 10) + 16
}
</script>
