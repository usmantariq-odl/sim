'use client'

import { Moon, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'
import { useEffect, useState } from 'react'
import { Button, Tooltip } from '@/components/emcn'

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Button
        variant='outline'
        className='h-[36px] w-[36px] rounded-[10px] border-[var(--border)] bg-gradient-to-br from-[var(--surface-2)] to-[var(--surface-3)] shadow-sm hover:shadow-md hover:scale-105 transition-all duration-200'
      >
        <Sun className='h-[18px] w-[18px] text-[var(--text-secondary)]' />
      </Button>
    )
  }

  const isDark = theme === 'dark' || theme === 'system'

  const handleToggle = () => {
    setTheme(isDark ? 'light' : 'dark')
  }

  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        <Button
          variant='outline'
          onClick={handleToggle}
          className='h-[36px] w-[36px] rounded-[10px] border-[var(--border)] bg-gradient-to-br from-[var(--surface-2)] to-[var(--surface-3)] shadow-sm hover:shadow-md hover:scale-105 transition-all duration-200'
        >
          {isDark ? (
            <Moon className='h-[18px] w-[18px] text-[var(--text-primary)] transition-all duration-300 rotate-0 hover:rotate-12' />
          ) : (
            <Sun className='h-[18px] w-[18px] text-[var(--text-primary)] transition-all duration-300 rotate-0 hover:rotate-90' />
          )}
        </Button>
      </Tooltip.Trigger>
      <Tooltip.Content>
        <p>Switch to {isDark ? 'light' : 'dark'} mode</p>
      </Tooltip.Content>
    </Tooltip.Root>
  )
}
