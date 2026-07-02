import json
import re
import sys

_MEMORY_OP_PATTERN = re.compile(
    r'<\|?(?:SEARCH|READ|TAIL|COUNT|APPEND|WRITE|CREATE|UPSERT_MAP|DONE)\|?>',
    re.IGNORECASE,
)

_OP_BLOCK_PATTERN = re.compile(
    r'<\|?(?:SEARCH|READ|TAIL|COUNT|APPEND|WRITE|CREATE|UPSERT_MAP)\|?>.*?<\|?END\|?[>}]',
    re.IGNORECASE | re.DOTALL,
)

_DONE_TAG_PATTERN = re.compile(r'<\|?DONE\|?>', re.IGNORECASE)

_ACTION_TAG_PATTERN = re.compile(
    r'<\|?ACTION\|?>.*?<\|?END\|?[>}]',
    re.IGNORECASE | re.DOTALL,
)

_CODE_BLOCK_WRAPPERS = [
    (re.compile(
        r'```[a-zA-Z]*[ \t]*\n[ \t]*'
        r'(<\|?(?:SEARCH|READ|TAIL|COUNT|APPEND|WRITE|CREATE|UPSERT_MAP|ACTION|DONE)\|?>.*?)'
        r'\n[ \t]*```',
        re.DOTALL | re.IGNORECASE,
    ), r'\1'),
    (re.compile(
        r'\n[ \t]*```[a-zA-Z]+[ \t]*\s*$',
    ), ''),
]

def _has_memory_ops(text):
    return bool(_MEMORY_OP_PATTERN.search(text))

def _is_action_only(text):
    stripped = _ACTION_TAG_PATTERN.sub('', text).strip()
    if _has_memory_ops(text):
        return False
    return len(stripped) < 200

def clean_format_artifacts(text):
    for pattern, replacement in _CODE_BLOCK_WRAPPERS:
        text = pattern.sub(replacement, text)
    return text

def trim_action_from_response(text):
    m = re.search(r'<\|?ACTION\|?>', text, re.IGNORECASE)
    if m:
        return text[:m.start()].strip()
    return text.strip()

def trim_post_op_prose(text):
    ends = [m.end() for m in _OP_BLOCK_PATTERN.finditer(text)] \
         + [m.end() for m in _DONE_TAG_PATTERN.finditer(text)]
    if not ends:
        return text, False
    last_end = max(ends)
    trimmed = text[:last_end].rstrip()
    return trimmed, trimmed != text.rstrip()

def postprocess_selected_data(data):
    stats = {
        "input_count": len(data),
        "format_cleaned": 0,
        "action_only_filtered": 0,
        "action_trimmed": 0,
        "prose_only_tail_filtered": 0,
        "post_op_prose_trimmed": 0,
        "fence_survived_filtered": 0,
        "output_count": 0,
    }

    result = []
    for item in data:
        conv = item.get("conversations", [])
        if not conv:
            continue

        cleaned = False
        for msg in conv:
            if msg.get("role") == "assistant":
                original = msg["content"]
                msg["content"] = clean_format_artifacts(original)
                if msg["content"] != original:
                    cleaned = True
        if cleaned:
            stats["format_cleaned"] += 1

        last_asst = conv[-1] if conv[-1].get("role") == "assistant" else None
        if last_asst is None:
            continue
        last_text = last_asst["content"]

        done_match = re.search(r'<\|?DONE\|?>', last_text)
        if done_match:
            truncated = last_text[:done_match.end()].strip()
            truncated = clean_format_artifacts(truncated)
            truncated, _ = trim_post_op_prose(truncated)
            if truncated != last_text.strip():
                stats["action_trimmed"] += 1
            last_asst["content"] = truncated
            result.append(item)
            continue

        has_action = bool(_ACTION_TAG_PATTERN.search(last_text))
        if has_action:
            if _is_action_only(last_text):
                stats["action_only_filtered"] += 1
                continue
            trimmed = trim_action_from_response(last_text)
            trimmed = clean_format_artifacts(trimmed)
            if not _has_memory_ops(trimmed):
                stats["prose_only_tail_filtered"] += 1
                continue
            trimmed, did_trim_tail = trim_post_op_prose(trimmed)
            if did_trim_tail:
                stats["post_op_prose_trimmed"] += 1
            last_asst["content"] = trimmed
            stats["action_trimmed"] += 1
            result.append(item)
            continue

        if _has_memory_ops(last_text):
            trimmed, did_trim_tail = trim_post_op_prose(last_text)
            if did_trim_tail:
                stats["post_op_prose_trimmed"] += 1
                last_asst["content"] = trimmed
            result.append(item)
        else:
            stats["action_only_filtered"] += 1

    defenced = []
    for item in result:
        if "```" in _last_assistant_text(item.get("conversations", [])):
            stats["fence_survived_filtered"] += 1
            continue
        defenced.append(item)
    result = defenced

    stats["output_count"] = len(result)
    return result, stats

def validate_conversation(conv):
    if not isinstance(conv, list) or len(conv) < 2:
        return False
    if conv[0].get("role") != "user":
        return False
    for i, msg in enumerate(conv):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected:
            return False
    if conv[-1].get("role") != "assistant":
        return False
    last = conv[-1].get("content", "")
    if not last.strip():
        return False
    return bool(
        re.search(r'<\|?DONE\|?>', last)
        or re.search(r'<\|?ACTION\|?>', last)
        or _has_memory_ops(last)
    )

def _last_assistant_text(conv):
    for msg in reversed(conv or []):
        if msg.get("role") == "assistant":
            return msg.get("content", "") or ""
    return ""

def verify_postprocess(data):
    checks = {
        "items": len(data),
        "no_op_last": 0,
        "plaintext_fence": 0,
        "action_in_last": 0,
        "post_op_tail": 0,
    }
    for item in data:
        last = _last_assistant_text(item.get("conversations", []))
        if not _has_memory_ops(last):
            checks["no_op_last"] += 1
        if "```" in last:
            checks["plaintext_fence"] += 1
        if re.search(r'<\|?ACTION\|?>', last, re.IGNORECASE):
            checks["action_in_last"] += 1
        ends = [m.end() for m in _OP_BLOCK_PATTERN.finditer(last)] \
             + [m.end() for m in _DONE_TAG_PATTERN.finditer(last)]
        if ends and len(last[max(ends):].strip()) > 0:
            checks["post_op_tail"] += 1
    return checks

def apply_postprocess(data):
    valid = [x for x in data if validate_conversation(x.get("conversations", []))]
    processed, pp_stats = postprocess_selected_data(valid)
    checks = verify_postprocess(processed)
    info = {
        "raw_count": len(data),
        "format_valid_count": len(valid),
        "postprocess": pp_stats,
        "sanitization": checks,
        "violations": sum(v for k, v in checks.items() if k != "items"),
    }
    return processed, info

def main(argv):
    if len(argv) != 3:
        sys.stderr.write(
            "usage: python postprocess.py IN.json OUT.json\n")
        return 2
    in_path, out_path = argv[1], argv[2]
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.stderr.write("ERROR: input is not a JSON list\n")
        return 3
    processed, info = apply_postprocess(data)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    print(json.dumps(info, indent=2))
    if info["violations"] > 0:
        sys.stderr.write(
            "FATAL: {} validity violation(s) after post-processing; "
            "refusing to certify this dataset.\n".format(
                info["violations"]))
        return 1
    if info["postprocess"]["output_count"] == 0:
        sys.stderr.write("FATAL: zero examples survived post-processing.\n")
        return 4
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
