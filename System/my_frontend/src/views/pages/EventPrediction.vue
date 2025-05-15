<template>
  <div class="p-6 space-y-6">
    <!-- 操作按钮 -->
    <div class="space-x-4">
      <button @click="showForm = !showForm" class="px-4 py-2 bg-blue-600 text-white rounded">
        {{ showForm ? '隐藏输入表单' : '显示输入表单' }}
      </button>
      <button @click="fillExample" class="px-4 py-2 bg-purple-600 text-white rounded">一键填充样例</button>
      <button @click="predict" class="px-4 py-2 bg-green-600 text-white rounded">开始预测</button>
      <button @click="showVisual = !showVisual" class="px-4 py-2 bg-pink-600 text-white rounded">可视化展示</button>
      <span class="text-gray-600 italic ml-4" v-if="exampleType">✔️ 已填充示例</span>
    </div>

    <!-- 输入表单区域 -->
    <div v-if="showForm" class="bg-white p-4 rounded shadow grid grid-cols-2 gap-6">
      <div>
        <h3 class="font-semibold text-lg mb-2">叙事事件</h3>
        <div v-for="(e, i) in narrativeEvents" :key="i" class="grid grid-cols-4 gap-2 mb-2">
          <input v-model="e.s" class="input" placeholder="s" />
          <input v-model="e.v" class="input" placeholder="v" />
          <input v-model="e.o" class="input" placeholder="o" />
          <input v-model="e.p" class="input" placeholder="p" />
        </div>
      </div>
      <div>
        <h3 class="font-semibold text-lg mb-2">候选事件</h3>
        <div v-for="(e, i) in candidateEvents" :key="i" class="grid grid-cols-4 gap-2 mb-2">
          <input v-model="e.s" class="input" placeholder="s" />
          <input v-model="e.v" class="input" placeholder="v" />
          <input v-model="e.o" class="input" placeholder="o" />
          <input v-model="e.p" class="input" placeholder="p" />
        </div>
      </div>
    </div>

    <!-- 可视化展示区 -->
    <div v-if="showVisual" class="grid grid-cols-2 gap-6">
      <div>
        <h3 class="font-bold text-lg mb-2">📚 叙事事件链</h3>
        <div class="flex flex-wrap gap-2">
          <template v-for="(e, i) in narrativeEvents">
            <div class="bg-blue-100 text-blue-800 px-3 py-2 rounded shadow font-medium">
              {{ formatEvent(e) }}
            </div>
            <div v-if="i < narrativeEvents.length - 1" class="text-xl">➡️</div>
          </template>
        </div>
      </div>

      <div>
        <h3 class="font-bold text-lg mb-2">🎯 候选事件卡片</h3>
        <div class="grid grid-cols-2 gap-3">
          <div
            v-for="(e, i) in candidateEvents"
            :key="i"
            class="bg-gray-100 border rounded shadow p-3"
          >
            <div class="font-semibold">候选事件：</div>
            <div class="text-gray-700 mt-1">{{ formatEvent(e) }}</div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="result" class="bg-white rounded p-4 shadow border mt-6">
      <h3 class="font-bold text-lg mb-3">预测结果</h3>
      <div v-if="formattedResult" class="prose max-w-none bg-white p-5 rounded shadow" v-html="formattedResult"></div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { marked } from 'marked'

const showForm = ref(true)
const showVisual = ref(false)
const narrativeEvents = ref(Array.from({ length: 8 }, () => ({ s: '', v: '', o: '', p: '' })))
const candidateEvents = ref(Array.from({ length: 5 }, () => ({ s: '', v: '', o: '', p: '' })))
const result = ref('')
const formattedResult = ref('')
const exampleType = ref('')

function formatEvent(e) {
  return [e.s, e.v, e.o, e.p].filter(Boolean).join(' ')
}

function fillExample() {
  const type = Math.random() > 0.5 ? 'us_ec' : 'us_politics'
  exampleType.value = type
if (type === 'us_politics') {
    narrativeEvents.value = [
      { s: 'Donald Trump', v: 'announced', o: 'candidacy', p: 'in Florida' },
      { s: 'Joe Biden', v: 'withdrew', o: 'from race', p: 'in July' },
      { s: 'Kamala Harris', v: 'secured', o: 'nomination', p: 'in August' },
      { s: 'Donald Trump', v: 'debated', o: 'Harris', p: 'in September' },
      { s: 'Donald Trump', v: 'won', o: 'election', p: 'in November' },
      { s: 'Joe Biden', v: 'conceded', o: 'defeat', p: 'after election' },
      { s: 'Kamala Harris', v: 'delivered', o: 'concession speech', p: 'at Howard University' },
      { s: 'Donald Trump', v: 'inaugurated', o: 'as president', p: 'in January' }
    ]
    candidateEvents.value = [
      { s: 'Donald Trump', v: 'appointed', o: 'cabinet', p: 'after inauguration' },
      { s: 'Kamala Harris', v: 'returned', o: 'to Senate', p: 'after election' },
      { s: 'Joe Biden', v: 'retired', o: 'from politics', p: 'in January' },
      { s: 'Donald Trump', v: 'signed', o: 'executive orders', p: 'in February' },
      { s: 'Kamala Harris', v: 'published', o: 'memoir', p: 'in March' }
    ]
  }else if(type === 'us_ec'){
    narrativeEvents.value = [
        { s: 'President Donald Trump', v: 'announced', o: 'sweeping tariffs on imports', p: '' },
        { s: 'The European Union', v: 'criticized', o: 'the U.S. tariffs', p: '' },
        { s: 'The European Union', v: 'threatened', o: 'countermeasures', p: '' },
        { s: 'Global markets', v: 'experienced', o: 'significant declines', p: '' },
        { s: 'Senator Ted Cruz', v: 'warned', o: 'tariffs could lead to a recession and political losses for Republicans', p: '' },
        { s: 'Canada', v: 'initiated', o: 'a dispute with the World Trade Organization over the U.S. car tariffs', p: '' },
        { s: 'Treasury Secretary Scott Bessent', v: 'defended', o: 'the tariffs', p: 'as necessary for economic recalibration' },
        { s: 'Protests', v: 'erupted', o: 'across the United States', p: 'opposing the tariff policies' }
    ];
    candidateEvents.value = [
        { s: 'President Trump', v: 'agreed', o: 'to lift certain tariffs', p: 'in exchange for favorable trade terms' },
        { s: 'The European Union', v: 'imposed', o: 'retaliatory tariffs on American goods', p: '' },
        { s: 'The stock market', v: 'rebounded', o: '', p: 'following positive trade negotiations' },
        { s: 'Senator Ted Cruz', v: 'introduced', o: 'legislation', p: 'to limit presidential tariff authority' },
        { s: 'Canada', v: 'withdrew', o: 'its WTO dispute', p: 'after reaching a bilateral agreement with the U.S.' }
    ];
  }
   else {
    narrativeEvents.value = [
      { s: 'friends', v: 'knew', o: 'simpson', p: '' },
      { s: 'simpson', v: 'became', o: 'famous', p: '' },
      { s: 'simpson', v: 're', o: '', p: '' },
      { s: 'simpson', v: 'killed', o: '', p: 'friday' },
      { s: 'simpson', v: 'writes', o: 'press', p: '' },
      { s: 'simpson', v: 'used', o: 'room', p: '' },
      { s: 'simpson', v: 'was', o: 'convicted', p: '' },
      { s: 'law', v: 'apply', o: 'simpson', p: '' }
    ]
    candidateEvents.value = [
      { s: 'simpson', v: 'come up', o: 'rule', p: '' },
      { s: 'simpson', v: 'invent', o: 'defens', p: '' },
      { s: 'simpson', v: 'circulated', o: '', p: '' },
      { s: 'simpson', v: 'attacks', o: 'deaths', p: '' },
      { s: 'simpson', v: 'pitching', o: '', p: '' }
    ]
  }
}

async function predict() {
  const events = narrativeEvents.value.map(e => ({ ...e }))
  const candidates = candidateEvents.value.map(e => ({ ...e }))
  try {
    const res = await axios.post('http://localhost:8000/predict/event', {
      events,
      candidates
    })
    const rawText = res.data.result || ''
    const safeText = rawText
      .replace(/<answer>/g, '<mark style="background: #fde68a; font-weight: bold;">')
      .replace(/<\/answer>/g, '</mark>')
    result.value = safeText
    formattedResult.value = marked.parse(safeText)
  } catch (e) {
    result.value = '❌ 请求失败: ' + e
    formattedResult.value = `<span style="color: red;">${e}</span>`
  }
}
</script>

<style scoped>
.input {
  @apply p-2 border rounded w-full;
}
</style>