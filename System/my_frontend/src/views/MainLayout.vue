<template>
  <div class="flex h-screen font-sans">
    <!-- 左侧导航 -->
    <aside class="w-64 bg-[#2c3e50] text-white flex flex-col p-4">
      <h2 class="text-xl font-bold mb-6">舆情事件预测系统</h2>
      <nav class="flex flex-col space-y-4 text-sm">
        
        <!-- 事件因果与行为心理图可视化 -->
        <div>
          <div class="px-3 py-2 text-xs font-semibold text-gray-300 uppercase">事件因果与行为心理图可视化</div>
          <router-link
            to="/system/entity-query"
            class="flex items-center gap-2 px-4 py-2 rounded hover:bg-[#34495e]"
            active-class="bg-[#34495e]"
          >
            🔍 实体信息检索
          </router-link>
          <router-link
            to="/system/graph"
            class="flex items-center gap-2 px-4 py-2 rounded hover:bg-[#34495e]"
            active-class="bg-[#34495e]"
          >
            🌐 完整图谱可视化
          </router-link>
        </div>

        <!-- 事件因果与行为心理图构建 -->
        <div>
          <div class="px-3 py-2 text-xs font-semibold text-gray-300 uppercase">事件因果与行为心理图构建</div>
          <router-link
            to="/system/causality"
            class="flex items-center gap-2 px-4 py-2 rounded hover:bg-[#34495e]"
            active-class="bg-[#34495e]"
          >
            🔁 因果识别与行为心理映射
          </router-link>
        </div>

        <!-- 事件预测平台 -->
        <div>
          <div class="px-3 py-2 text-xs font-semibold text-gray-300 uppercase">事件预测平台</div>
          <router-link
            to="/system/predict"
            class="flex items-center gap-2 px-4 py-2 rounded hover:bg-[#34495e]"
            active-class="bg-[#34495e]"
          >
            🧠 事件预测
          </router-link>
        </div>

      </nav>
    </aside>

    <!-- 主体内容 -->
    <div class="flex-1 flex flex-col">
      <!-- 顶部栏 -->
      <header class="bg-white shadow border-b px-6 py-4 text-lg font-medium text-gray-700">
        舆情事件预测系统
      </header>

      <!-- 内容视图 -->
      <main class="flex-1 bg-gray-100 p-8 overflow-auto">
        <router-view />
      </main>
    </div>
  </div>
</template>
