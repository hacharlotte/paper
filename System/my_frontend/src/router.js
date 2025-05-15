// src/router.js
import { createRouter, createWebHistory } from 'vue-router'
import Welcome from './views/Welcome.vue'
import MainLayout from './views/MainLayout.vue'
import EntityQuery from './views/pages/EntityQuery.vue'
import KGView from './views/pages/KGView.vue'
import CausalityPredict from './views/pages/CausalityPredict.vue'  // ✅ 事件因果预测页面
import EventPrediction from './views/pages/EventPrediction.vue'// ✅ 事件预测页面

const routes = [
  { path: '/', component: Welcome },
  {
    path: '/system',   // 所有功能走 system 路由
    component: MainLayout,
    children: [
      { path: 'entity-query', component: EntityQuery },       // 实体信息检索
      { path: 'graph', component: KGView },                   // 图谱可视化
      { path: 'causality', component: CausalityPredict },      // 事件因果预测
      { path: 'predict', component: EventPrediction }      // 事件预测
    ]
  }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
