# LoCoMo：补充数据候选

状态：已核验官方 README、授权和实际 JSON；不是全文精读。

- 当前官方 locomo10 数据有 10 个长对话、1986 个问题；不能与较早 50 对话版本混称。
- 提供对话日期、dia_id 和 QA evidence，但本次发现 4 题无 evidence，9 题不能直接将全部 evidence 精确匹配到轮次。
- 错误格式和缺失引用保留清单，不静默修正；纯文本实验需要明确图像信息缺失。
- 根许可证为 CC BY-NC 4.0，需署名并遵守非商业限制。

选择：作为补充，先建立可靠映射，再做与主数据不同的多会话场景验证。

来源：https://github.com/snap-research/locomo 。锁定版本与原始校验见 configs/dataset_sources.json 和 reports/dataset_audit.json。
