<template>
  <div class="p-4">
    <h2 class="text-xl font-bold mb-4">完整图谱可视化</h2>
    <div id="viz" style="height: 700px; border: 1px solid #ccc;"></div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import axios from 'axios'
import { Network } from 'vis-network/standalone'

onMounted(async () => {
  const container = document.getElementById('viz')
  if (!container) {
    console.error('❌ 容器未找到')
    return
  }

  try {
    const res = await axios.get('http://localhost:8000/graph')
    const raw = res.data

    // ✅ 与 entity-query 样式统一：颜色区分不同类型
    const levelColors = {
      Entity: '#6FA8DC',       // 天蓝色
      Event: '#FFFF00',        // 金黄色
      Sentiment: '#FF6F61'     // 略深红色（模拟 entity-query 中的红）
    }

    const coloredNodes = raw.nodes.map(node => {
      let color = '#ccc'
      let title = node.title || node.label || 'Unnamed'

      if (node.label === 'Entity') {
        color = levelColors.Entity
      } else if (node.label === 'Event') {
        color = levelColors.Event
      } else if (node.label === 'Sentiment') {
        color = levelColors.Sentiment
      }

  return {
    id: node.id,
    label: title,   // 🟢 设置为 title，才会显示原始句子文本
    title: title,
    color
  }
    })

    const options = {
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
    }

    new Network(container, { nodes: coloredNodes, edges: raw.edges }, options)
  } catch (err) {
    console.error('❌ 获取图数据失败:', err)
  }
})
</script>
