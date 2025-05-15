<!-- GraphView.vue -->
<template>
  <div class="w-full h-full p-4">
    <h1 class="text-2xl font-bold mb-4">图谱可视化</h1>
    <div id="viz" class="w-full h-[700px] bg-white border rounded-xl shadow"></div>
  </div>
</template>

<script setup>
import { nextTick, defineExpose } from 'vue'
import NeoVis from 'neovis.js'

let viz = null

// 暴露方法供父组件或路由组件调用
const renderGraph = async () => {
  await nextTick()
  requestAnimationFrame(() => {
    const config = {
      container_id: "viz",
      server_url: "bolt://127.0.0.1:7687",
      server_user: "neo4j",
      server_password: "1195955206",
      labels: {
        Entity: { caption: "label", shape: "ellipse" },
        Event: { caption: "label", shape: "dot" },
        Sentiment: { caption: "label", shape: "box" }
      },
      relationships: {
        PARTICIPATE_IN: { caption: "type" },
        CAUSES: { caption: "type" },
        HAS_SENTIMENT: { caption: "type" }
      },
      initial_cypher: `
        MATCH (n)-[r]->(m)
        RETURN n, r, m LIMIT 200
      `,
      arrows: true
    }

    viz = new NeoVis(config)
    viz.registerOnEvent("completed", () => console.log("✅ 图谱渲染完成"))
    viz.registerOnEvent("error", (e) => console.error("❌ 渲染失败", e))

    console.log("🚀 图谱开始渲染")
    viz.render()
  })
}

defineExpose({ renderGraph })
</script>
