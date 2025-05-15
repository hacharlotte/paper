<!-- frontend/src/components/GraphView.vue -->
<template>
  <div id="viz" class="w-full h-[500px] border mt-4"></div>
</template>

<script setup>
import { onMounted } from 'vue'
import NeoVis from 'neovis.js'

onMounted(() => {
  const config = {
    containerId: 'viz',
    neo4j: {
      serverUrl: 'bolt://localhost:7687',
      serverUser: 'neo4j',
      serverPassword: 'password'
    },
    labels: {
      Entity: { caption: 'name' },
      Event: { caption: 'label' }
    },
    relationships: {
      CAUSES: { caption: true }
    },
    initialCypher: 'MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50'
  }

  const viz = new NeoVis(config)
  viz.render()
})
</script>
