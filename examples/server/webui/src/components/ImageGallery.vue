<script setup lang="ts">
import { ref, onMounted, nextTick, computed } from 'vue'
import { Modal, Carousel } from 'bootstrap'
import { useGenerationStore } from '@/stores/generation'
import { useRouter } from 'vue-router'

interface HistoryItem {
  name: string
  params?: any
}

const store = useGenerationStore()
const router = useRouter()

const images = ref<HistoryItem[]>([])
const isLoading = ref(true)
const error = ref<string | null>(null)

const modalElement = ref<HTMLElement | null>(null)
const carouselElement = ref<HTMLElement | null>(null)
let modalInstance: Modal | null = null
let carouselInstance: Carousel | null = null

const activeIndex = ref(0)

// Gallery Layout State
const columnsPerRow = ref(Number(localStorage.getItem('gallery-columns')) || 4)

const setColumns = (count: number) => {
  columnsPerRow.value = Math.max(2, Math.min(12, count))
  localStorage.setItem('gallery-columns', String(columnsPerRow.value))
}

// Expose for parent component
defineExpose({ columnsPerRow, setColumns })

async function fetchImages() {
  isLoading.value = true
  error.value = null
  try {
    const response = await fetch('/v1/history/images')
    if (!response.ok) {
      throw new Error('Failed to fetch image history from the server.')
    }
    const data = await response.json()
    images.value = data

    // Wait for the DOM to update with the new images
    await nextTick()
    
    // Initialize modal and carousel now that the elements are in the DOM
    if (modalElement.value && carouselElement.value) {
      modalInstance = new Modal(modalElement.value)
      carouselInstance = new Carousel(carouselElement.value, {
        interval: false, // Do not auto-cycle
        touch: true,
      })

      // Track active index
      carouselElement.value.addEventListener('slid.bs.carousel', (event: any) => {
        activeIndex.value = event.to
      })
    }

  } catch (e: any) {
    error.value = e.message
  } finally {
    isLoading.value = false
  }
}

function openModal(index: number) {
  if (carouselInstance && modalInstance) {
    activeIndex.value = index
    carouselInstance.to(index)
    modalInstance.show()
  }
}

function reuseParameters(navigate = true) {
  const item = images.value[activeIndex.value]
  if (item && item.params) {
    const p = item.params
    if (p.prompt) store.prompt = p.prompt
    if (p.negative_prompt) store.negativePrompt = p.negative_prompt
    if (p.sample_steps) store.steps = p.sample_steps
    if (p.seed !== undefined) store.seed = p.seed
    if (p.cfg_scale !== undefined) store.cfgScale = p.cfg_scale
    if (p.sampling_method) {
      // Try to find matching sampler in our list
      const sm = p.sampling_method.toLowerCase().replace('_a', ' a')
      if (store.samplers.includes(sm)) {
        store.sampler = sm
      }
    }
    if (p.width) store.width = p.width
    if (p.height) store.height = p.height
    
    if (navigate) {
      // Close modal and navigate
      modalInstance?.hide()
      
      if (p.is_img2img || p.init_image) {
        router.push('/img2img')
      } else {
        router.push('/')
      }
    }
  }
}

async function sendToImg2Img() {
  const item = images.value[activeIndex.value]
  if (!item) return
  
  const imageUrl = '/outputs/' + item.name
  
  try {
    const response = await fetch(imageUrl)
    const blob = await response.blob()
    const reader = new FileReader()
    reader.onloadend = () => {
      store.initImage = reader.result as string
      
      // Load dimensions
      const img = new Image()
      img.onload = () => {
         // Optionally we could auto-set width/height here too
      }
      img.src = store.initImage

      reuseParameters(false) // Reuse parameters but don't navigate yet
      modalInstance?.hide()
      router.push('/img2img')
    }
    reader.readAsDataURL(blob)
  } catch (err) {
    console.error('Failed to fetch image for img2img:', err)
  }
}

onMounted(() => {
  fetchImages()
})
</script>

<template>
  <div>
    <!-- Loading/Error/Empty State -->
    <div v-if="isLoading" class="text-center my-5">
      <div class="spinner-border text-primary" role="status">
        <span class="visually-hidden">Loading...</span>
      </div>
    </div>
    <div v-else-if="error" class="alert alert-danger">
      {{ error }}
    </div>
    <div v-else-if="images.length === 0" class="text-center text-muted my-5">
      <i class="bi bi-image-fill display-1"></i>
      <p class="mt-3">No images found in history.</p>
      <p>Generate some images with the "Save Images Automatically" setting enabled.</p>
    </div>

    <!-- Image Grid (Custom CSS Grid) -->
    <div v-else class="custom-gallery-grid" :style="{ '--cols': columnsPerRow }">
      <div v-for="(image, index) in images" :key="image.name" class="gallery-item">
        <div class="card card-clickable shadow-sm h-100 border-0 bg-dark bg-opacity-10" @click="openModal(index)">
          <img :src="'/outputs/' + image.name" class="card-img-top" :alt="image.name" loading="lazy" />
          <div class="card-footer p-1 text-truncate x-small text-muted text-center border-0 bg-transparent">
            {{ image.name.substring(4, 14) }}
          </div>
        </div>
      </div>
    </div>

    <!-- Modal -->
    <div class="modal fade" ref="modalElement" tabindex="-1" aria-labelledby="imageModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-xl modal-dialog-centered">
        <div class="modal-content shadow-lg">
          <div class="modal-header">
            <h5 class="modal-title" id="imageModalLabel">
              {{ images[activeIndex]?.name || 'Image Viewer' }}
            </h5>
            <div class="ms-auto me-2 d-flex gap-2">
              <button 
                class="btn btn-outline-success btn-sm"
                @click="sendToImg2Img"
              >
                <i class="bi bi-image"></i> Send to Img2Img
              </button>
              <button 
                v-if="images[activeIndex]?.params" 
                class="btn btn-outline-primary btn-sm"
                @click="reuseParameters(true)"
              >
                <i class="bi bi-arrow-repeat"></i> Reuse Parameters
              </button>
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body p-0 bg-black">
            <!-- Carousel -->
            <div ref="carouselElement" id="imageHistoryCarousel" class="carousel slide">
              <div class="carousel-inner">
                <div v-for="(image, index) in images" :key="`carousel-${image.name}`" class="carousel-item" :class="{ active: index === activeIndex }">
                  <img :src="'/outputs/' + image.name" class="d-block w-100" :alt="image.name">
                </div>
              </div>
              <button class="carousel-control-prev" type="button" data-bs-target="#imageHistoryCarousel" data-bs-slide="prev">
                <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                <span class="visually-hidden">Previous</span>
              </button>
              <button class="carousel-control-next" type="button" data-bs-target="#imageHistoryCarousel" data-bs-slide="next">
                <span class="carousel-control-next-icon" aria-hidden="true"></span>
                <span class="visually-hidden">Next</span>
              </button>
            </div>
          </div>
          <div class="modal-footer justify-content-start" v-if="images[activeIndex]?.params">
             <div class="small w-100 text-muted overflow-auto" style="max-height: 100px;">
                <strong>Prompt:</strong> {{ images[activeIndex].params.prompt }}<br>
                <strong>Seed:</strong> {{ images[activeIndex].params.seed }} |
                <strong>Steps:</strong> {{ images[activeIndex].params.sample_steps }} |
                <strong>CFG:</strong> {{ images[activeIndex].params.cfg_scale }} |
                <strong>Sampler:</strong> {{ images[activeIndex].params.sampling_method }} |
                <strong>Size:</strong> {{ images[activeIndex].params.width }}x{{ images[activeIndex].params.height }}
             </div>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>

<style scoped>
.custom-gallery-grid {
  display: grid;
  grid-template-columns: repeat(var(--cols, 4), 1fr);
  gap: 1rem;
}

.gallery-item {
  min-width: 0; /* Prevent grid breakout */
}

.card-img-top {
  aspect-ratio: 1 / 1;
  object-fit: cover;
}

.x-small {
  font-size: 0.65rem;
}
.card-clickable {
  cursor: pointer;
  transition: transform 0.2s ease-in-out;
}
.card-clickable:hover {
  transform: scale(1.03);
}
.modal-body img {
  max-height: 80vh;
  object-fit: contain;
}
</style>
