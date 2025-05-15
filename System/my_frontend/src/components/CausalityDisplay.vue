<template>
  <div class="flex space-x-4">

    <!-- 左侧：气泡 -->
    <div class="w-1/2 space-y-2 overflow-y-auto max-h-[80vh] border rounded p-2 bg-white">
      <div v-for="pair in data" :key="pair.id" class="border p-2 rounded shadow-sm">
        <p class="font-semibold mb-1 text-sm text-gray-600">事件对: ({{pair.i}}, {{pair.j}})</p>
        <div class="flex space-x-2">
          <div class="bg-indigo-100 p-2 flex-1 rounded">{{formatEvent(pair.e1)}}</div>
          <div class="bg-yellow-100 p-2 flex-1 rounded">{{formatEvent(pair.e2)}}</div>
        </div>
      </div>
    </div>

    <!-- 右侧：图谱 -->
    <div class="w-1/2 border rounded p-2 bg-white">
      <div ref="graphContainer" class="h-[80vh] w-full"></div>
    </div>
      <!-- 事件详情卡片 -->
      <div
        v-if="selectedEvent"
        class="absolute top-2 right-2 bg-white border shadow-md rounded p-4 w-[300px]"
      >
        <h4 class="text-sm font-semibold mb-2">事件详情</h4>
        <p class="text-sm">主语: {{ selectedEvent.s || '-' }}</p>
        <p class="text-sm">谓语: {{ selectedEvent.v || '-' }}</p>
        <p class="text-sm">宾语: {{ selectedEvent.o || '-' }}</p>
        <p class="text-sm">介宾: {{ selectedEvent.p || '-' }}</p>
      </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { Network } from 'vis-network'

const props = defineProps({ data: Array })
const graphContainer = ref(null)
const selectedEvent = ref(null)

function formatEvent(e) {
  return `(${e.s || '-'}, ${e.v || '-'}, ${e.o || '-'}, ${e.p || '-'})`
}

function renderGraph() {
  const nodes = [], edges = []
  props.data.forEach(pair => {
    nodes.push({ id: pair.i, label: pair.e1.v, event: pair.e1 })
    nodes.push({ id: pair.j, label: pair.e2.v, event: pair.e2 })
    edges.push({ from: pair.i, to: pair.j, label: "CAUSES" })
  })

  const uniqNodes = Object.values(Object.fromEntries(nodes.map(n => [n.id, n])))
  const network = new Network(graphContainer.value, { nodes: uniqNodes, edges }, {
    interaction: { hover: true },
    nodes: {
  shape: 'dot',
  size: 20,
  color: {
    background: '#FFFF00',
    border: '#FFA500'
  },
  font: {
    color: '#333',
    size: 14
  }
},
    edges: { arrows: 'to', color: '#888', font: { align: 'top' } }
  })

  network.on("click", function (params) {
    if (params.nodes.length > 0) {
      const nodeId = params.nodes[0]
      const node = uniqNodes.find(n => n.id === nodeId)
      selectedEvent.value = node?.event || null
    } else {
      selectedEvent.value = null
    }
  })
}

watch(() => props.data, renderGraph)
onMounted(renderGraph)
</script>

