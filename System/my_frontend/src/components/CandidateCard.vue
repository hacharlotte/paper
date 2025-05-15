<template>
  <div class="border rounded-lg shadow bg-white p-4 hover:shadow-lg transition">
    <p class="text-gray-800 text-sm">
      <span class="font-semibold">候选事件：</span><br />
      {{ [event.s, event.v, event.o, event.p].filter(x => x).join(' ') }}
    </p>
  </div>
</template>

<script setup>
defineProps({ event: Object })
</script>
