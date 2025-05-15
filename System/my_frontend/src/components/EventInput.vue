<!-- frontend/src/components/EventInput.vue -->
<template>
  <div class="flex flex-col gap-4">
    <textarea
      v-model="text"
      class="w-full h-32 border border-gray-300 p-2"
      placeholder="请输入要识别的文本"
    ></textarea>
    <button @click="submitText" class="bg-blue-600 text-white p-2 rounded hover:bg-blue-700">
      开始识别因果关系
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const text = ref('')
const emit = defineEmits(['result'])

async function submitText() {
  const res = await axios.post('http://localhost:8000/predict/causality', {
    input_text: text.value
  })
  emit('result', res.data)
}
</script>
