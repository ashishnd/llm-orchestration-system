/** Empty string = same-origin (Vite dev proxy or nginx in Docker). */
export const API_BASE = import.meta.env.VITE_API_BASE ?? ''
