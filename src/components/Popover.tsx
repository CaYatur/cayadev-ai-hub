import { useEffect, useRef, type ReactNode } from "react";

/**
 * A menu anchored above its trigger.
 *
 * Opens upward because every use sits on the composer at the bottom of the
 * window. Closes on outside click and on Escape — a menu that traps the user is
 * worse than no menu.
 */
export function Popover({
  open,
  onClose,
  children,
  align = "left",
  width,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  align?: "left" | "right";
  width?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    function onPointer(e: MouseEvent) {
      // The trigger handles its own toggle, so ignore clicks that land on it.
      const target = e.target as Node;
      if (ref.current && !ref.current.parentElement?.contains(target)) onClose();
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }

    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={ref}
      className={`popover popover-${align}`}
      style={width ? { width } : undefined}
      role="menu"
    >
      {children}
    </div>
  );
}

export function PopoverLabel({ children }: { children: ReactNode }) {
  return <div className="popover-label">{children}</div>;
}

export function PopoverItem({
  label,
  hint,
  selected,
  shortcut,
  badge,
  danger,
  onClick,
}: {
  label: string;
  hint?: string;
  selected?: boolean;
  shortcut?: string;
  badge?: string;
  danger?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={`popover-item${selected ? " popover-item-on" : ""}${danger ? " popover-item-danger" : ""}`}
      onClick={onClick}
      role="menuitem"
    >
      <span className="popover-item-main">
        <span className="popover-item-label">
          {label}
          {badge && <span className="popover-badge">{badge}</span>}
        </span>
        {hint && <span className="popover-item-hint">{hint}</span>}
      </span>
      {selected && <span className="popover-check">✓</span>}
      {shortcut && <span className="popover-shortcut">{shortcut}</span>}
    </button>
  );
}

export function PopoverDivider() {
  return <div className="popover-divider" />;
}
