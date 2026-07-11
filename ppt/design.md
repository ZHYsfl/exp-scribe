我们的PPT用slidev做，写md就能用slidev转化成ppt,整体用中文，术语用英文。字体、排版、风格高度模仿这个PPT /root/autodl-tmp/ppt/examples。PPT不应是文字的堆砌，而是内容的有效组织排版，以方便我进行进一步的展示，但也不能追求形式主义，内容和形式要找到平衡，我觉得这个PPT /root/autodl-tmp/ppt/examples的这点也值得我们学习。

首先是封面，还是用cover.jpg当背景,/root/autodl-tmp/ppt/cover.jpg
封面的文字格式也是重点学习 /root/autodl-tmp/ppt/examples的。关于封面的主题，就是：SCRIBE : Loop Engineering in Long Horizon Tasks Needs Agent to Write What They Did Each Turn然后标注一些我的个人信息 周浩洋，JLU,SE，这个项目的RA日期是从4.17开始一直持续到现在以及之后等。

你可以先通读一下我们的工作：/root/autodl-tmp/README.md

---

然后就是我们工作的介绍了，第一部分：

01 training free 方法-不可能实现loop engineering的token效率提升 这个是每页中上位置放的字，
起两页分别放：
/root/autodl-tmp/ppt/pics/training-free-token效率不可能提升-1.png
/root/autodl-tmp/ppt/pics/training-free-token效率不可能提升-2.png

---

然后第二部分：
02 training方法-使用react进行loop engineering
这个是每页中上位置放的字，起一页放：
/root/autodl-tmp/ppt/pics/broad-react-training-method.png

---

然后第三部分：
03 training方法-使用react进行loop engineering的问题
这个是每页中上位置放的字，起一页放：
/root/autodl-tmp/ppt/pics/broad-react-training-method-problems.png

---

然后第四部分：
04 training方法-使用SCRIBE进行loop engineering的问题
这个是每页中上位置放的字，起一页放：
/root/autodl-tmp/ppt/pics/SCRIBE-training-method.png

---

然后第五部分：

05 training方法-细述SCRIBE进行loop engineering的上下文管理机制
这个是每页中上位置放的字，
然后你把那个/root/autodl-tmp/README.md里那个最长的例子搬过来，每页这样放：每页只放左边一个turn，右边一个turn，中间一个箭头。而且你还要用颜色展示出前后kv-cache复用的部分（公共前缀）！
你可以用颜色巧妙展示变化之类的，尽你全力发挥，让机制展示得更清晰！
我们一共最终到了turn11，所以一共这个例子占了turn0->turn1,turn1->turn2,....,turn10->turn11一共11页。
然后你这部分的第12页，你总结下我们的上下文机制，/root/autodl-tmp/README.md这里面也用文字说明的很清楚了，你可以直接照抄，这一块你不用担心内容和形式要找到平衡，这块注重的是上下文机制定义的严谨性重内容的严谨性全面性。
---

然后第六部分，

06 training方法-the reward mechanism of a turn
这部分只起一页，总结这16个metric凑一起的权重配比（看代码）及设计哲学。16个metric的逐条详细说明见/root/autodl-tmp/README.md，不再单独成页（单页总览已足够展示）。
你可以用颜色或其他技巧巧妙展示让机制展示得更清晰！

---

然后是第七部分，我们的贡献（可以直接当A会比如ICLR的paper的）摘要（这部分用英文）：

07 abstract of our contribution:
这个是每页中上位置放的字，
起一页写摘要即可。
请你仔细阅读/root/autodl-tmp/README.md，摘要核心是围绕着scribe一个小而美的改动和这个改动衍生出的上下文机制的变化，导致了非常多衍生出来的好处：1.无痛扩展agent上下文，大部分时间实现自动无耗时压缩2.实现了token-level级别的信用分配，grpo训练效率比broad react高3.实现了对一个loop engineering中的task的准确性和token效率上的提升，并且这两个指标比训练前和用react设计模式训练都高4.kv cache复用率高，等。
---

最后谢谢聆听，然后Q&A