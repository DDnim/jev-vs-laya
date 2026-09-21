"""Task routing benchmark: pick the LLM tier and reasoning effort for a task card.
Cards are synthetic (modelled on a personal Kanban), mixed zh / ja / en like the real thing."""

MODELS = {
  "flash":  "cheapest/fastest model: trivial edits, one-line CSS, quick lookups, greetings, tests",
  "sonnet": "mid-tier coding model: routine feature or bug fix with a clear spec, one or two files",
  "opus":   "strong model: multi-file features, debugging unknown causes, research plus deployment",
  "fable":  "top reasoning model: architecture, evaluating alternatives, writing specs, should-we decisions",
}
EFFORT = ["low: a single obvious step", "medium: routine work with a clear spec", "high: investigation or multi-file change", "xhigh: design or architecture decision"]

CASES = [
 dict(id="r01", card="KanbanのTocc列の幅を60%広げる", model="flash", effort=0),
 dict(id="r02", card="say test", model="flash", effort=0),
 dict(id="r03", card="你是哪个模型", model="flash", effort=0),
 dict(id="r04", card="README のタイポを直す（'recieve' → 'receive'）", model="flash", effort=0),
 dict(id="r05", card="Kanban完了チェック・Done移動に効果音を追加する。既存の settings に音量トグルを足す", model="sonnet", effort=1),
 dict(id="r06", card="ss-issueの動作を確認する: 起票パネルからカードが作られ、見出しが揃っているか", model="sonnet", effort=1),
 dict(id="r07", card="移动端添加开屏画面，降低初始化等待感。SwiftUI 的 SplashView，2 秒后进入主界面", model="sonnet", effort=1),
 dict(id="r08", card="日本の不動産売却手続きを調査して手順を Wiki にまとめる", model="sonnet", effort=1),
 dict(id="r09", card="Kanbanで画像を貼れない原因調査と修正。クリップボード経由だと落ちる、ドラッグは動く", model="opus", effort=2),
 dict(id="r10", card="主页を Cloudflare Workers にデプロイ。ビルド設定・DNS・環境変数を含めて本番まで通す", model="opus", effort=2),
 dict(id="r11", card="移动端：有序粘贴队列 —— 并发转录按录音提交顺序输出（修复乱序；长音频异步）", model="opus", effort=2),
 dict(id="r12", card="帮我创建一个自动同步 GitHub project 的 issue 到本地 kanban 的功能，基于 obsidian 插件", model="opus", effort=2),
 dict(id="r13", card="kanban的antigravity cli貌似用不了，你看一下", model="opus", effort=2),
 dict(id="r14", card="KanbanをObsidian Basesから独立させるべきか。依存の棚卸しと移行コストの比較", model="fable", effort=3),
 dict(id="r15", card="Kanban改善v2: 外置執行システム5原則で機能設計", model="fable", effort=3),
 dict(id="r16", card="完了チェックの動的生成と子カード化を設計する", model="fable", effort=3),
 dict(id="r17", card="重做不依赖Base的看板插件：先写 spec 和 ADR，再决定是否动手", model="fable", effort=3),
 dict(id="r18", card="Inkfall 云路径模型改由服务端决定：停发 pp_model / reasoning_effort 并隐藏云端模型选择（mobile + server 两边）", model="opus", effort=2),
 dict(id="r19", card="test4", model="flash", effort=0),
 dict(id="r20", card="mattpocock/skills を Kanban 運用に組み込む。どのスキルを・どの列で・誰が呼ぶかの運用設計", model="fable", effort=3),
]

QUESTIONS = {
 "model": {"type": "choice", "instructions": "Which LLM tier should handle the task in `card`?", "criteria": MODELS},
 "effort": {"type": "score", "instructions": "How much reasoning effort does the task in `card` need?", "criteria": EFFORT},
}
