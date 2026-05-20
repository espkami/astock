// =============================================================================
// THEME TOGGLE
// Supports: system preference, manual toggle, localStorage persistence
// =============================================================================

;(() => {
  const STORAGE_KEY = 'obvious-theme'

  function getPreferredTheme() {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) return stored
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }

  function updateButtonIcons(theme) {
    document.querySelectorAll('.theme-toggle').forEach((btn) => {
      btn.textContent = theme === 'dark' ? '☀️' : '🌙'
      btn.title = theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'
    })
  }

  function setTheme(theme) {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
      root.classList.remove('light')
    } else {
      root.classList.add('light')
      root.classList.remove('dark')
    }
    localStorage.setItem(STORAGE_KEY, theme)
    updateButtonIcons(theme)
  }

  function toggleTheme() {
    const current = document.documentElement.classList.contains('dark') ? 'dark' : 'light'
    setTheme(current === 'dark' ? 'light' : 'dark')
  }

  // 立即应用主题（防止页面闪白/闪黑）
  const initialTheme = getPreferredTheme()
  const root = document.documentElement
  if (initialTheme === 'dark') {
    root.classList.add('dark')
    root.classList.remove('light')
  } else {
    root.classList.add('light')
    root.classList.remove('dark')
  }

  // DOM 加载后绑定按钮并更新图标
  document.addEventListener('DOMContentLoaded', () => {
    updateButtonIcons(getPreferredTheme())
    document.querySelectorAll('.theme-toggle').forEach((btn) => {
      btn.addEventListener('click', toggleTheme)
    })
  })

  // 监听系统主题变化（仅未手动设置时生效）
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem(STORAGE_KEY)) {
      setTheme(e.matches ? 'dark' : 'light')
    }
  })

  window.toggleTheme = toggleTheme
  window.setTheme = setTheme
})()
