<template>
  <div class="flex flex-wrap items-center">
    <template v-for="(e, i) in events">
      <div :key="'e' + i" class="flex items-center">
        <div class="bg-blue-100 text-blue-700 px-3 py-1 rounded shadow text-sm">
          {{ [e.s, e.v, e.o, e.p].filter(x => x).join(' ') }}
        </div>
        <div v-if="i !== events.length - 1" class="mx-2 text-xl text-gray-400">➔</div>
      </div>
    </template>
  </div>
</template>

<script setup>
defineProps({ events: Array })
</script>
