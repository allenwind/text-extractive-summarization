import numpy as np

template = "\033[38;5;{value}m{string}\033[0m"
def render_cmd_blue(text):
    return template.format(string=text, value=32)

def render_cmd_green(text):
    return template.format(string=text, value=43)

def render_cmd_red(text):
    return template.format(string=text, value=160)

md_template = '<font color="{hexcode}">{string}</font>'
def render_md_blue(text):
    return md_template.format(hexcode="#0000ff", string=text)

def render_md_green(text):
    return md_template.format(hexcode="#00ff00", string=text)

def render_md_red(text):
    return md_template.format(hexcode="#f20000", string=text)

def compute_match_subsequence(text1, text2, a_tag, b_tag):
    """高亮两个文本的最长公共子序列（LCS）"""
    if not (text1 and text2):
        return
    s1 = len(text1)
    s2 = len(text2)
    dp = np.zeros((s1 + 1, s2 + 1), dtype=np.int32)
    for i in range(1, s1 + 1):
        for j in range(1, s2 + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    maxlen = dp[-1][-1]

    i = s1 - 1
    j = s2 - 1
    locs = []
    while len(locs) < maxlen:
        if text1[i] == text2[j]:
            locs.append((i, j))
            i -= 1
            j -= 1
        elif dp[i + 1, j] > dp[i, j + 1]:
            j -= 1
        else:
            i -= 1
    locs[::-1] = locs

    sw1 = np.zeros(s1)
    sw2 = np.zeros(s2)
    ts1 = set([i for i, _ in locs])
    ts2 = set([j for _, j in locs])
    for i, j in enumerate(text1):
        if i in ts1:
            sw1[i] = a_tag
    for i, j in enumerate(text2):
        if i in ts2:
            sw2[i] = b_tag
    return sw1, sw2

def show_case(content, title, ptitle):
    """
    在LCS下，颜色标注规则：
    - content中的字出现在title中标绿色，同时title标注绿色
    - content中的字出现在ptitle中标蓝色，同时ptitle标注蓝色
    - content中的字同时出现在title与ptitle中，标注为红色
    """
    c_w1, t_w = compute_match_subsequence(content, title, a_tag=1, b_tag=1)
    c_w2, p_w = compute_match_subsequence(content, ptitle, a_tag=2, b_tag=2)
    c_w = c_w1 + c_w2
    rendered_content = ""
    for s, w in zip(content, c_w):
        if w == 1:
            rendered_content += render_cmd_green(s)
        elif w == 2:
            rendered_content += render_cmd_blue(s)
        elif w == 3:
            rendered_content += render_cmd_red(s)
        else:
            rendered_content += s

    rendered_title = ""
    for s, w in zip(title, t_w):
        if w == 1:
            rendered_title += render_cmd_green(s)
        else:
            rendered_title += s

    rendered_ptitle = ""
    for s, w in zip(ptitle, p_w):
        if w == 2:
            rendered_ptitle += render_cmd_blue(s)
        else:
            rendered_ptitle += s

    print("文章内容：")
    print(rendered_content, end="\n\n")
    print("真实title：", rendered_title)
    print("预测title：", rendered_ptitle)
    print("=" * 20)

if __name__ == "__main__":
    # for testing
    import string

    text = string.ascii_letters
    print(render_cmd_red(text))
    print(render_cmd_blue(text))
    print(render_cmd_green(text))

    with open("visualization.md", "w") as fd:
        fd.write(render_md_blue(text))
        fd.write(render_md_green(text))
        fd.write(render_md_red(text))

    content = "abcdef"
    title = "zzzabcd"
    ptitle = "cdefxxy"
    show_case(content, title, ptitle)
