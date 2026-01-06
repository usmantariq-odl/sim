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
        className='h-[32px] w-[32px] rounded-[8px] border-[var(--border)] bg-transparent hover:bg-[var(--surface-6)] dark:hover:bg-[var(--surface-5)]'
      >
        <Sun className='h-[16px] w-[16px] text-[var(--text-tertiary)]' />
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
          className='h-[32px] w-[32px] rounded-[8px] border-[var(--border)] bg-transparent hover:bg-[var(--surface-6)] dark:hover:bg-[var(--surface-5)]'
        >
          {isDark ? (
            <Moon className='h-[16px] w-[16px] text-[var(--text-secondary)] transition-transform duration-200 hover:scale-110' />
          ) : (
            <Sun className='h-[16px] w-[16px] text-[var(--text-secondary)] transition-transform duration-200 hover:scale-110' />
          )}
        </Button>
      </Tooltip.Trigger>
      <Tooltip.Content>
        <p>Switch to {isDark ? 'light' : 'dark'} mode</p>
      </Tooltip.Content>
    </Tooltip.Root>
  )
}
