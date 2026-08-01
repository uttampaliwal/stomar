/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_STOMAR_API_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

interface Window {
  __STOMAR_API_KEY__?: string
}
