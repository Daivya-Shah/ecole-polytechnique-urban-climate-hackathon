<template>
  <div>
    <header class="app-header">
      <h1 class="app-title">GenHack4</h1>
    </header>
    

    <div class="container">
      <div v-if="selectedCity && !loadingMetrics">
        <!-- TABS -->
        <div class="tabs">
          <button 
            class="tab" 
            :class="{ active: activeTab === 'overview' }" 
            @click="activeTab = 'overview'"
          >
            Overview
          </button>
          <button 
            class="tab" 
            :class="{ active: activeTab === 'charts' }" 
            @click="activeTab = 'charts'"
          >
            Interactive Charts
          </button>
          <button 
            class="tab" 
            :class="{ active: activeTab === 'ml' }" 
            @click="activeTab = 'ml'"
          >
            ML Models
          </button>
          <button 
            class="tab" 
            :class="{ active: activeTab === 'stations' }" 
            @click="activeTab = 'stations'"
          >
            Stations
          </button>
          <button 
            class="tab" 
            :class="{ active: activeTab === 'multimap' }" 
            @click="activeTab = 'multimap'"
          >
            Multi-City Map
          </button>
          <button 
            class="tab" 
            :class="{ active: activeTab === 'knn' }" 
            @click="activeTab = 'knn'"
          >
            KNN
          </button>

          <button 
            class="tab" 
            :class="{ active: activeTab === 'legend' }" 
            @click="activeTab = 'legend'"
          >
            Legend
          </button>
        </div>

        <!-- TAB CONTENT -->
        <div v-if="activeTab === 'overview'">
          <MetricsOverview :metrics="metrics" :city="selectedCity" />
        </div>

        <div v-if="activeTab === 'charts'">
          <InteractivePlots :city="selectedCity" :metrics="metrics" />
        </div>

        <div v-if="activeTab === 'ml'">
          <MLModels :city="selectedCity" />
        </div>


        <div v-if="activeTab === 'stations'">
          <StationDiscovery />
        </div>

        <div v-if="activeTab === 'multimap'">
          <MultiCityMap :availableCities="cities" />
        </div>

        <div v-if="activeTab === 'knn'">
          <KNNAnalysis :availableCities="cities" />
        </div>


        <keep-alive>
          <LegendTable v-if="activeTab === 'legend'" :city="selectedCity" />
        </keep-alive>
      </div>

      <div v-else-if="loadingMetrics" class="loading">
        <div class="spinner"></div>
        <span>Loading data...</span>
      </div>

      <HomePage v-if="!selectedCity" @start="scrollToSelector" />

      <div class="city-selector-anchor"></div>
      
      <CitySelector
        v-if="selectedCity || cities.length"
        :cities="cities"
        :selectedCity="selectedCity"
        @select="selectCity"
        :loading="loadingCities"
      />
    </div>
  </div>

</template>

<script>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import CitySelector from './components/CitySelector.vue'
import MetricsOverview from './components/MetricsOverview.vue'
import InteractivePlots from './components/InteractivePlots.vue'
import MLModels from './components/MLModels.vue'
import LegendTable from './components/LegendTable.vue'
import ThemeToggle from './components/ThemeToggle.vue'
import HomePage from './components/HomePage.vue'
import StationDiscovery from './components/StationDiscovery.vue'
import MultiCityMap from './components/MultiCityMap.vue'
import KNNAnalysis from './components/KNNAnalysis.vue'

const API_BASE = 'http://localhost:5000/api'

export default {
  name: 'App',
  components: {
    CitySelector,
    MetricsOverview,
    InteractivePlots,
    MLModels,
    LegendTable,
    ThemeToggle,
    HomePage,
    StationDiscovery,
    MultiCityMap,
    KNNAnalysis
  },
  setup() {
    const theme = ref('dark')
    const cities = ref([])
    const selectedCity = ref(null)
    const metrics = ref({})
    const features = ref({})
    const loadingCities = ref(true)
    const loadingMetrics = ref(false)
    const activeTab = ref('overview')
    const stationMetadata = ref([])

    const toggleTheme = () => {
      theme.value = theme.value === 'dark' ? 'light' : 'dark'
      localStorage.setItem('theme', theme.value)
      document.documentElement.setAttribute('data-theme', theme.value)
    }


    const scrollToSelector = () => {
      const selector = document.querySelector('.city-selector-anchor')
      if (selector) {
        selector.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }

    const fetchCities = async () => {
      try {
        loadingCities.value = true
        const response = await axios.get(`${API_BASE}/cities`)
        cities.value = response.data.cities
      } catch (error) {
        console.error('Error fetching cities:', error)
      } finally {
        loadingCities.value = false
      }
    }

    const fetchStationMetadata = async () => {
      try {
        const response = await axios.get(`${API_BASE}/stations`)
        stationMetadata.value = response.data
        console.log('[INFO] Loaded station metadata:', stationMetadata.value.length, 'stations')
      } catch (error) {
        console.error('Error fetching station metadata:', error)
      }
    }

    const selectCity = async (city) => {
      selectedCity.value = city
      activeTab.value = 'overview'
      await loadCityData(city)
    }

    const loadCityData = async (city) => {
  try {
    loadingMetrics.value = true
    const [metricsRes, featuresRes] = await Promise.all([
      axios.get(`${API_BASE}/metrics/${city}`),
      axios.get(`${API_BASE}/features/${city}`)
    ])
    
    // ✅ Handle NaN values in JSON strings
    let metricsData = metricsRes.data
    if (typeof metricsData === 'string') {
      // Replace NaN with null before parsing
      metricsData = metricsData.replace(/:\s*NaN/g, ': null')
      metrics.value = JSON.parse(metricsData)
    } else {
      metrics.value = metricsData
    }
    
    features.value = featuresRes.data
    
  } catch (error) {
    console.error('Error loading city data:', error)
    metrics.value = {}
    features.value = {}
  } finally {
    loadingMetrics.value = false
  }
}



onMounted(() => {
  const savedTheme = localStorage.getItem('theme')
  if (savedTheme) {
    theme.value = savedTheme
  }
  // Apply theme to HTML element
  document.documentElement.setAttribute('data-theme', theme.value)
  
  fetchCities()
  fetchStationMetadata()
})


    return {
      theme,
      cities,
      selectedCity,
      metrics,
      features,
      loadingCities,
      loadingMetrics,
      activeTab,
      stationMetadata,
      scrollToSelector,
      toggleTheme,
      selectCity
    }
  }
}
</script>

<style>
/* Global styles remain in src/assets/styles.css */
</style>
