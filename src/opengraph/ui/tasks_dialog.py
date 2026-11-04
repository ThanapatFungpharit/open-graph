"""The Tasks panel: symbolic computations with their state, running time and Cancel."""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft

from opengraph import events
from opengraph.tasks import Task, TaskManager, TaskState, format_duration
from opengraph.ui import runtime

TICK_MS = 100  # running times refresh this often while something runs


@dataclass(slots=True)
class _Item:
    """One task's line in the list."""

    title: ft.Text
    status: ft.Text
    time: ft.Text
    cancel: ft.TextButton | None
    row: ft.Row


class TasksDialog:
    """Lists every task of a :class:`TaskManager`, newest first; non-modal."""

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager
        self.visible = False
        self._items: dict[int, _Item] = {}
        self.list = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, height=220)
        self.summary = ft.Text("", size=12, expand=True)
        self.cancel_all_button = ft.TextButton(
            "Cancel All", on_click=lambda _: manager.cancel_all()
        )
        self.clear_button = ft.TextButton(
            "Clear Finished", on_click=lambda _: manager.clear_finished()
        )
        header = ft.Row(
            [
                ft.Text("Task", expand=True, weight=ft.FontWeight.W_600, size=12),
                ft.Text("Status", width=150, weight=ft.FontWeight.W_600, size=12),
                ft.Text("Time", width=60, weight=ft.FontWeight.W_600, size=12),
                ft.Container(width=70),
            ]
        )
        self.control = ft.Card(
            ft.Container(
                ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Text("Tasks", weight=ft.FontWeight.BOLD, size=15, expand=True),
                                ft.IconButton(
                                    ft.Icons.CLOSE,
                                    icon_size=16,
                                    tooltip="Close",
                                    on_click=lambda _: self.close(),
                                ),
                            ]
                        ),
                        header,
                        self.list,
                        ft.Row([self.summary, self.cancel_all_button, self.clear_button]),
                    ],
                    spacing=6,
                    tight=True,
                ),
                padding=14,
                width=560,
            ),
            elevation=6,
            visible=False,
        )
        self._timer = events.Timer(TICK_MS)
        self._timer.timeout.connect(self._tick)
        manager.changed.connect(self.refresh)
        self.refresh()

    def set_compact(self, compact: bool) -> None:
        """On a narrow screen the panel spans the window."""
        content = self.control.content
        assert isinstance(content, ft.Container)
        content.width = None if compact else 560
        self.control.left = 8 if compact else None
        self.control.right, self.control.bottom = (8, 8) if compact else (12, 12)
        runtime.push(self.control)

    def show(self) -> None:
        self.visible = self.control.visible = True
        self.refresh()

    def close(self) -> None:
        self.visible = self.control.visible = False
        runtime.push(self.control)

    def refresh(self) -> None:
        """Bring the list in line with the manager's tasks."""
        tasks = self.manager.tasks()
        ids = {t.id for t in tasks}
        for task_id in [i for i in self._items if i not in ids]:
            del self._items[task_id]
        for task in tasks:
            item = self._items.get(task.id)
            if item is None:
                title = ft.Text(task.title, expand=True, size=12, tooltip=task.title, no_wrap=True)
                status = ft.Text("", width=150, size=12)
                time = ft.Text("", width=60, size=12, text_align=ft.TextAlign.RIGHT)
                row = ft.Row([title, status, time], spacing=6)
                item = self._items[task.id] = _Item(title, status, time, None, row)
            self._update(item, task)
        # Newest first.
        self.list.controls = [self._items[t.id].row for t in reversed(tasks)]
        active = self.manager.active()
        running = sum(t.state is TaskState.RUNNING for t in active)
        queued = len(active) - running
        self.summary.value = f"{running} running, {queued} queued" if active else "No tasks running"
        self.cancel_all_button.disabled = not active
        self.clear_button.disabled = len(active) == len(tasks)
        if active and not self._timer.is_active():
            self._timer.start()
        elif not active:
            self._timer.stop()
        runtime.push(self.control)

    def _update(self, item: _Item, task: Task) -> None:
        status = task.state.value
        if task.state is TaskState.FAILED and task.detail:
            status = f"{status}: {task.detail}"
        item.status.value = status
        item.status.tooltip = status
        item.time.value = format_duration(task.elapsed())
        if task.finished:
            if item.cancel is not None:
                item.cancel = None
                item.row.controls = [item.title, item.status, item.time]
        elif item.cancel is None:
            item.cancel = ft.TextButton(
                "Cancel",
                tooltip="Stop this computation",
                width=70,
                on_click=lambda _, i=task.id: self.manager.cancel(i),
            )
            item.row.controls = [item.title, item.status, item.time, item.cancel]

    def _tick(self) -> None:
        for task in self.manager.active():
            item = self._items.get(task.id)
            if item is not None:
                item.time.value = format_duration(task.elapsed())
        runtime.push(self.list)

    def cancel_button(self, task_id: int) -> ft.TextButton | None:
        """The Cancel button of a task's row, while it has one."""
        item = self._items.get(task_id)
        return item.cancel if item is not None else None

    def cancel(self, task_id: int) -> None:
        """Click a task's Cancel button."""
        if self.cancel_button(task_id) is not None:
            self.manager.cancel(task_id)

    @property
    def row_count(self) -> int:
        return len(self.list.controls)
