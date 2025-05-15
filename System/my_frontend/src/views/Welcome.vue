<template>
  <div
    class="min-h-screen flex items-center justify-center bg-cover bg-center"
    style="background-image: url('/images/背景.png');"
  >
    <div
      class="text-center bg-white/50 backdrop-blur-lg p-8 md:p-10 rounded-2xl shadow-2xl border border-gray-200"
    >
      <h1 class="text-3xl md:text-4xl font-extrabold text-gray-900 mb-6">
        欢迎进入舆情事件预测系统
      </h1>
      <button
        @click="enterSystem"
        class="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-transform transform hover:scale-105 duration-300"
      >
        进入系统
      </button>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
const router = useRouter()
const enterSystem = () => router.push('/system/entity-query')
</script>
