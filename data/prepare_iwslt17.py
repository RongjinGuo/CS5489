"""
Prepare IWSLT17 TED en-zh data:

- 读取 data/en-zh/ 下面的原始文件：
    - train.tags.en-zh.en / train.tags.en-zh.zh
    - IWSLT17.TED.dev2010.en-zh.en.xml / .zh.xml
    - IWSLT17.TED.tst20xx.en-zh.en.xml / .zh.xml

- 输出到：
    data/iwslt17/train/train.en
    data/iwslt17/train/train.zh
    data/iwslt17/valid/valid.en
    data/iwslt17/valid/valid.zh
    data/iwslt17/test/test.en
    data/iwslt17/test/test.zh
"""

from pathlib import Path
import xml.etree.ElementTree as ET


# --------- 基础工具函数 ---------

def read_xml_sentences(xml_path: Path):
    """从 IWSLT XML 文件中读取所有 <seg> 句子"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sents = []
    for seg in root.iter("seg"):
        text = (seg.text or "").strip()
        if text:
            sents.append(text)
    return sents


def write_split(out_dir: Path, split_name: str, en_sents, zh_sents):
    """把某个 split 写成 .en / .zh 两个文件"""
    out_split = out_dir / split_name
    out_split.mkdir(parents=True, exist_ok=True)

    with open(out_split / f"{split_name}.en", "w", encoding="utf-8") as f_en:
        for s in en_sents:
            f_en.write(s + "\n")

    with open(out_split / f"{split_name}.zh", "w", encoding="utf-8") as f_zh:
        for s in zh_sents:
            f_zh.write(s + "\n")

    print(f"Saved {split_name} to {out_split}  ({len(en_sents)} pairs)")


# --------- 读各个 split ---------

def read_train_from_tags(base_dir: Path):
    """读取 train.tags.en-zh.en / train.tags.en-zh.zh 作为训练集"""
    en_path = base_dir / "train.tags.en-zh.en"
    zh_path = base_dir / "train.tags.en-zh.zh"

    if not en_path.exists() or not zh_path.exists():
        print("[train] train.tags.* not found, train split will be empty.")
        return [], []

    def read_tags(path: Path):
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过 <url>, <talkid>, <title>, <description> 等标签行
                if not line or line.startswith("<"):
                    continue
                lines.append(line)
        return lines

    en_sents = read_tags(en_path)
    zh_sents = read_tags(zh_path)

    assert len(en_sents) == len(zh_sents), "train en/zh length mismatch"
    print(f"[train] collected {len(en_sents)} parallel sentences")
    return en_sents, zh_sents


def read_xml_split(base_dir: Path, prefix: str):
    """
    读取 dev / tst：
      prefix = "dev" or "tst"
      匹配: IWSLT17.TED.{prefix}*.en-zh.en.xml / .zh.xml
    """
    en_files = sorted(base_dir.glob(f"IWSLT17.TED.{prefix}*.en-zh.en.xml"))
    zh_files = sorted(base_dir.glob(f"IWSLT17.TED.{prefix}*.en-zh.zh.xml"))

    print(f"[{prefix}] found en_files={len(en_files)}, zh_files={len(zh_files)}")

    assert len(en_files) == len(zh_files), f"{prefix}: en/zh file count mismatch"

    all_en, all_zh = [], []
    for en_f in en_files:
        zh_f = base_dir / en_f.name.replace(".en-zh.en.xml", ".en-zh.zh.xml")
        assert zh_f.exists(), f"Missing zh file for {en_f.name}"

        en_sents = read_xml_sentences(en_f)
        zh_sents = read_xml_sentences(zh_f)
        assert len(en_sents) == len(zh_sents), f"{en_f.name}: en/zh length mismatch"

        all_en.extend(en_sents)
        all_zh.extend(zh_sents)

    print(f"[{prefix}] collected {len(all_en)} parallel sentences")
    return all_en, all_zh


# --------- main ---------

def main():
    # 你的 IWSLT17 原始文件放在这里：
    base_dir = Path("data/en-zh")   # ★★★ 就是你 find 出来的那个目录

    out_dir = Path("data/iwslt17")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 训练集：train.tags.*
    train_en, train_zh = read_train_from_tags(base_dir)
    write_split(out_dir, "train", train_en, train_zh)

    # 2) 验证集：dev2010 xml
    valid_en, valid_zh = read_xml_split(base_dir, "dev2010")
    # 为了和后面的调用兼容，split 名字还是 "valid"
    write_split(out_dir, "valid", valid_en, valid_zh)

    # 3) 测试集：所有 tst20xx xml 合在一起
    test_en, test_zh = read_xml_split(base_dir, "tst")
    write_split(out_dir, "test", test_en, test_zh)

    print("\nDone. Plain text data written to data/iwslt17/")


if __name__ == "__main__":
    main()