<template>
  <div class="bg-white p-6 rounded-lg shadow-md">
    <h2 class="text-xl font-semibold text-gray-800 mb-4">事件因果与行为心理图构建</h2>

    <!-- 文本输入 -->
    <div class="mb-4">
      <textarea
        v-model="inputText"
        class="w-full p-3 border rounded focus:outline-none focus:ring focus:ring-blue-400"
        rows="5"
        placeholder="请输入文本"
        @keydown.up="handleUpArrow"
      ></textarea>
    </div>

    <!-- 操作按钮 -->
    <div class="mb-6 space-x-4">
      <button
        @click="extractEvents"
        class="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
      >
        抽取事件
      </button>

      <button
        @click="predictCausality"
        :disabled="!events.length"
        :class="['px-4 py-2 rounded', events.length ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-gray-300 text-gray-500 cursor-not-allowed']"
      >
        事件因果识别
      </button>

      <button
        @click="predictSentiment"
        :disabled="!events.length || !pairs.length"
        :class="['px-4 py-2 rounded', events.length && pairs.length ? 'bg-emerald-600 hover:bg-emerald-700 text-white' : 'bg-gray-300 text-gray-500 cursor-not-allowed']"
      >
        行为心理映射
      </button>
    </div>

    <!-- 抽取结果展示 -->
    <div v-if="events.length" class="grid grid-cols-2 gap-4">
      <!-- 左侧：事件列表 -->
      <div class="flex-1 min-h-[500px]">
        <h3 class="text-lg font-semibold mb-2">事件列表</h3>
        <ul class="space-y-2">
          <li
            v-for="(e, i) in paginatedEvents"
            :key="(currentPage - 1) * pageSize + i"
            class="bg-white p-2 rounded border shadow-sm"
          >
            <span class="text-gray-700 font-medium">事件{{ (currentPage - 1) * pageSize + i + 1 }}：</span>
            <span class="text-gray-800">
              ({{ e.event.s || '_' }}, {{ e.event.v || '_' }}, {{ e.event.o || '_' }}, {{ e.event.p || '_' }})
            </span>
          </li>
        </ul>
        <div class="mt-3 flex justify-between items-center text-sm text-gray-600">
          <button
            @click="currentPage--"
            :disabled="currentPage === 1"
            class="px-2 py-1 border rounded disabled:opacity-50"
          >
            上一页
          </button>
          <span>第 {{ currentPage }} / {{ totalPages }} 页</span>
          <button
            @click="currentPage++"
            :disabled="currentPage === totalPages"
            class="px-2 py-1 border rounded disabled:opacity-50"
          >
            下一页
          </button>
        </div>

      </div>

      <!-- 右侧：图谱（展示事件，没有边）-->
      <div class="flex-1 min-h-[500px] bg-gray-50 p-3 rounded shadow">
        <div ref="eventGraphContainer" class="w-full h-full border rounded bg-white"></div>
      </div>
    </div>

    <!-- 因果对图谱 -->
    <div v-if="pairs.length" class="mt-8">
      <h3 class="text-lg font-semibold mb-2">因果事件对展示</h3>
      <CausalityDisplay :data="pairs" />
    </div>
  </div>

  <!-- 情感视图展示 -->
  <div v-if="outputJson.length" class="mt-10 grid grid-cols-2 gap-4">
    <!-- 左侧：文本 + 情感事件展示 -->
    <div class="space-y-4">
      <h3 class="text-lg font-semibold">句子与事件心理标签</h3>
      <div
        v-for="(entry, idx) in outputJson"
        :key="idx"
        class="p-3 border rounded bg-white shadow"
      >
        <p class="text-gray-700 mb-2">📄 {{ entry.text }}</p>
        <ul class="text-sm space-y-1">
          <li v-for="(e, i) in entry.events" :key="i" class="text-gray-600">
            → ({{ e.s || '_' }}, {{ e.v || '_' }}, {{ e.o || '_' }}, {{ e.p || '_' }}) -
            <span
              :class="{
                'text-green-600': e.sentiment === 'positive',
                'text-red-600': e.sentiment === 'negative',
                'text-gray-500': e.sentiment === 'neutral'
              }"
            >
              {{ e.sentiment || 'unknown' }}
            </span>
          </li>
        </ul>
      </div>
    </div>

    <!-- 右侧：图谱展示 -->
    <div class="bg-gray-50 rounded shadow p-3">
      <h3 class="text-lg font-semibold mb-2 text-gray-700">事件行为心理图</h3>
      <div ref="sentimentGraphContainer" class="h-[500px] border rounded bg-white"></div>
    </div>
  </div>

</template>

<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'
import axios from 'axios'
import CausalityDisplay from '../../components/CausalityDisplay.vue'
import { Network } from 'vis-network'

const currentPage = ref(1)
const pageSize = 8
import { computed } from 'vue'

const outputJson = ref([])
const eventsWithSentiment = ref([])
const new_pairs = ref([])

async function predictSentiment() {
  const res = await axios.post('http://localhost:8000/predict/sentiment', {
    input_text: inputText.value
  })

  outputJson.value = res.data.output_json || []
  eventsWithSentiment.value = res.data.enriched_events || []
  new_pairs.value = res.data.causal_pairs || []
  await nextTick()
  renderSentimentGraph()
}

const paginatedEvents = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return events.value.slice(start, start + pageSize)
})

const totalPages = computed(() => Math.ceil(events.value.length / pageSize))

const inputText = ref('')
const events = ref([])
const pairs = ref([])
const sentences = ref([])

const eventGraphContainer = ref(null)

async function extractEvents() {
  const res = await axios.post('http://localhost:8000/extract/events', {
    input_text: inputText.value
  })
  events.value = res.data.events || []
  sentences.value = res.data.sentences || []

  await nextTick()
  renderEventGraph()
}

async function predictCausality() {
  const res = await axios.post('http://localhost:8000/predict/causality', {
    input_text: inputText.value
  })
  pairs.value = res.data.pairs || res.data.causal_pairs?.data || []
}

function renderEventGraph() {
  const nodes = events.value.map((e, i) => ({
    id: i,
    label: e.event.v || '?',
    title: `(${e.event.s || '_'}, ${e.event.v || '_'}, ${e.event.o || '_'}, ${e.event.p || '_'})`
  }))
  const edges = []

  new Network(eventGraphContainer.value, { nodes, edges }, {
    layout: { improvedLayout: true },
    nodes: { shape: 'dot', size: 20, color: { background: '#FFFF00', border: '#FFA500' }, font: { color: '#333', size: 14 } },
    physics: { enabled: true, solver: 'forceAtlas2Based' }
  })
}

const sentimentGraphContainer = ref(null)

function renderSentimentGraph() {
  const nodes = []
  const edges = []

  const eventIndexMap = new Map()

  eventsWithSentiment.value.forEach((e, idx) => {
    const nodeId = idx
    const label = e.event.v || '?'
    const title = `(${e.event.s || '_'}, ${e.event.v || '_'}, ${e.event.o || '_'}, ${e.event.p || '_'})`

    nodes.push({
      id: nodeId,
      label,
      title,
      shape: 'dot',
      size: 20,
      color: { background: '#FFFF00', border: '#FFA500' },
      font: { color: '#1f2937' }
    })

    const key = JSON.stringify([e.event.s, e.event.v, e.event.o, e.event.p, e.sentence_id, e.trigger_index])
    eventIndexMap.set(key, nodeId)

    const sentiment = e.event.sentiment
    if (sentiment) {
      const sentimentNodeId = `sentiment-${idx}`

      nodes.push({
        id: sentimentNodeId,
        label: sentiment,
        shape: 'dot',
        font: { color: '#1f2937', size: 14 },
        color: { background: sentiment === 'positive' ? '#bbf7d0' : sentiment === 'negative' ? '#fecaca' : '#e5e7eb', border: '#6b7280' }
      })

      edges.push({ from: nodeId, to: sentimentNodeId, arrows: 'to', label: '情感', font: { align: 'top' }, color: { color: '#a3a3a3' } })
    }
  })

  for (const pair of new_pairs.value) {
    const key1 = JSON.stringify([pair.e1.s, pair.e1.v, pair.e1.o, pair.e1.p, pair.e1.sentence_id, pair.e1.trigger_index])
    const key2 = JSON.stringify([pair.e2.s, pair.e2.v, pair.e2.o, pair.e2.p, pair.e2.sentence_id, pair.e2.trigger_index])

    const from = eventIndexMap.get(key1)
    const to = eventIndexMap.get(key2)

    if (from !== undefined && to !== undefined) {
      edges.push({ from, to, arrows: 'to', label: '因果', font: { align: 'middle' }, color: { color: '#6b7280' } })
    } else {
      console.warn('❗未匹配因果边:', key1, '->', key2)
    }
  }

  new Network(
    sentimentGraphContainer.value,
    { nodes, edges },
    {
      layout: { improvedLayout: true },
      nodes: { font: { size: 14, color: '#374151' } },
      edges: { smooth: true },
      physics: { enabled: true, solver: 'forceAtlas2Based' }
    }
  )
}

// 处理“上箭头”按键事件
const predefinedSamples = ref([
  'On April 8, 2025, President Trump intensified the trade conflict with China by imposing a 104% tariff on Chinese goods. This action came after China failed to meet his demand to withdraw retaliatory tariffs on American products. The announcement caused volatility in U.S. financial markets, with stocks sliding into negative territory. Trump defended the policy, claiming it generates $2 billion per day for the U.S. economy, and asserted that China will eventually agree to a trade deal. In response, China condemned the move as "blackmail" and pledged to resist. ',
]);

function handleUpArrow(event) {
  if (event.key === 'ArrowUp') {
    const currentIndex = predefinedSamples.value.indexOf(inputText.value)
    const nextIndex = (currentIndex - 1 + predefinedSamples.value.length) % predefinedSamples.value.length
    inputText.value = predefinedSamples.value[nextIndex]
  }
}

</script>
