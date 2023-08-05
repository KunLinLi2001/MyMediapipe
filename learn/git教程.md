<article class="baidu_pl">
        <div id="article_content" class="article_content clearfix">
        <link rel="stylesheet" href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/ck_htmledit_views-b3c43d3711.css">
                <div id="content_views" class="markdown_views prism-atom-one-light">
                    <svg xmlns="http://www.w3.org/2000/svg" style="display: none;">
                        <path stroke-linecap="round" d="M5,0 0,2.5 5,5z" id="raphael-marker-block" style="-webkit-tap-highlight-color: rgba(0, 0, 0, 0);"></path>
                    </svg>
                    <p>我们先理清<strong>Git和Github的区别</strong>，Git是个版本控制的工具，用来管理本地的代码工程，它可以记录代码内容的变更；而<a href="https://so.csdn.net/so/search?q=Github&spm=1001.2101.3001.7020" target="_blank" class="hl hl-1" data-report-click="{"spm":"1001.2101.3001.7020","dest":"https://so.csdn.net/so/search?q=Github&spm=1001.2101.3001.7020","extra":"{\"searchword\":\"Github\"}"}" data-tit="Github" data-pretit="github">Github</a>是一个代码托管平台，我们可以使用Git将本地代码上传到Github。</p> 
<p>那<strong>为什么要学Git</strong>，上面说到Git是一个版本控制工具，它可以记录代码内容的变更，方便我们对项目的管理，它主要有以下的用途：</p> 
<ul><li> <p><strong>代码备份</strong></p> <p>举一个简单的例子，写好一个程序后，想要再加一个新功能，添加这个新功能需要对很多个文件进行改动。但是添加完这个功能后发现，这个新功能是个坑，然后就会想着回退到上一个版本，这时候如果有备份还好，没有备份的话就得<code>ctrl+z</code>，甚至要按照自己的记忆恢复以前的版本</p> <p>如果使用Git就好办了，直接使用git切换下版本或者分支就可以了。</p> <p>当然先备份一下原始代码，然后在上面改，也是一个很简单且有效的方法，但是随着新功能的增加，程序所占用的内存越来越大，而且每个版本间的区别很难直观看出，这不利于我们对项目的管理</p> </li><li> <p><strong>多人协作</strong></p> <p>假设要做一个项目，这个项目中有多个功能，大多数情况下，我们不可能一个功能一个功能地做，而是进行团队分工，多个功能同步进行，这时候如果没有使用Git进行版本控制，那么在将这些功能统合时将会出现很大的问题</p> </li><li> <p><strong>以后工作的需要</strong></p> <p>目前很多公司都在用Git进行项目管理，学会Git对以后的工作还是很有好处的</p> </li></ul> 
<p>本文将介绍Git的基本工作流程，以及这些流程对应的命令行指令，最后对<a href="https://so.csdn.net/so/search?q=VScode&spm=1001.2101.3001.7020" target="_blank" class="hl hl-1" data-report-click="{"spm":"1001.2101.3001.7020","dest":"https://so.csdn.net/so/search?q=VScode&spm=1001.2101.3001.7020","extra":"{\"searchword\":\"VScode\"}"}" data-tit="VScode" data-pretit="vscode">VScode</a>中内置的Git进行介绍和演示</p> 
<h2><a name="t0"></a><a id="Git_22"></a>Git安装</h2> 
<p>Git在Windows上和在Linux上的使用方式基本无差别，所以本文中的演示主要在Windows上进行</p> 
<h3><a name="t1"></a><a id="Windows_26"></a>Windows</h3> 
<p>在<a href="https://git-scm.com/downloads">这里</a>可以下载最新版的Git，然后一路Next即可</p> 
<h3><a name="t2"></a><a id="Ubuntu_30"></a>Ubuntu</h3> 
<pre data-index="0" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">sudo</span> <span class="token function">apt-get</span> <span class="token function">install</span> <span class="token function">git</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<h2><a name="t3"></a><a id="Git_36"></a>Git工作流程</h2> 
<p>基本工作流程图：</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859404.png" alt="02-概念-02"> 
</center>
<ul><li>工作区(Workspace)：平时存放项目代码的地方</li><li>暂存区(Index/Stage)：用于临时存放改动信息</li><li>本地仓库(Repository)：存放所有提交的版本数据</li><li>远程仓库(Remote)：托管代码的服务器，比如我们经常用的Github就是个代码托管平台</li></ul> 
<p>git的基本工作流程如下：</p> 
<ol><li>在工作区中添加、修改文件</li><li>将工作区中需要进行版本管理的文件放入暂存区</li><li>将暂存区的文件提交到git本地仓库</li><li>(<strong>optional</strong>)将本地仓库推送到远程仓库</li></ol> 
<p>为了方便以后的学习和工作，不建议直接使用GUI来操作Git，下面将针对上面的工作流程介绍一些常用的Git命令行指令，这些指令是比较简单的，敲熟练之后再上手GUI版本的Git就相当容易了</p> 
<h3><a name="t4"></a><a id="_56"></a>初始化</h3> 
<p>初始化git，有两种方式：</p> 
<pre data-index="1" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token comment"># 方式一:本地生成一个git</span>
<span class="token function">git</span> init
<span class="token comment"># 方式二:从远端克隆一个仓库</span>
<span class="token function">git</span> clone https://gitee.com/xxxxxx/xx.git
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li></ul></pre> 
<p>基本配置</p> 
<pre data-index="2" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token comment"># 配置用户名</span>
<span class="token function">git</span> config --global user.name <span class="token string">"name"</span>
<span class="token comment"># 配置邮箱</span>
<span class="token function">git</span> config --global user.email <span class="token string">"name@mail.com"</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li><li style="color: rgb(153, 153, 153);">4</li></ul></pre> 
<p>删除远程仓库</p> 
<pre data-index="3" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> remote <span class="token function">rm</span> origin
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<p>添加远程仓库</p> 
<pre data-index="4" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> remote <span class="token function">add</span> origin https://gitee.com/xxxxxx/xx.git
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<h3><a name="t5"></a><a id="_88"></a>推送至远程仓库</h3> 
<p>将已修改文件添加至暂存区</p> 
<pre data-index="5" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> <span class="token function">add</span> dir/filename <span class="token comment"># 添加指定文件</span>
<span class="token function">git</span> <span class="token function">add</span> <span class="token builtin class-name">.</span> <span class="token comment"># 添加所有已修改文件</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li></ul></pre> 
<p>将暂存区的改动提交到本地的版本库，使用<code>git commit</code>命令我们就会在本地版本库生成一个40位的哈希值，用于版本回退</p> 
<pre data-index="6" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> commit -m <span class="token string">"message"</span> <span class="token comment"># message就是本次提交的简要说明</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<p>本地上传，注意在推送前需要先从远程拉取</p> 
<pre data-index="7" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> push -u origin master <span class="token comment"># master可以更换为其他分支</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<h3><a name="t6"></a><a id="_109"></a>从远程仓库拉取</h3> 
<p>更新本地：</p> 
<pre data-index="8" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> pull origin master <span class="token comment"># master可以更换为其他分支</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<h3><a name="t7"></a><a id="_117"></a>分支管理</h3> 
<p>分支管理是版本控制中一个很重要的内容，在Git中主要有切换/创建分支(checkout)、合并分支(merge)两个指令</p> 
<p>下面是部分分支操作的指令和图示，圆圈○表示一个提交(commit)记录，矩形表示分支，它指向一个提交记录，由这个记录可以遍历之前所有的提交记录</p> 
<p>首先初始化了一个git，这个git中只有一个<code>master</code>分支，包含两个commit记录</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859294.png" alt="03-图示-01"> 
</center>
<p>现在我们创建一个新分支，命名为<code>develop</code>：</p> 
<pre data-index="9" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> checkout -b develop <span class="token comment"># 表示创建并切换到develop分支</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859295.png" alt="03-图示-02"> 
</center>
<p>此时<code>master</code>分支和<code>develop</code>分支都指向C1这个提交记录。我们分别在这两个分支上进行修改并提交：</p> 
<pre data-index="10" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> commit
<span class="token function">git</span> checkout master <span class="token comment"># 切换到master分支</span>
<span class="token function">git</span> commit
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li><li style="color: rgb(153, 153, 153);">3</li></ul></pre> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859353.png" alt="03-图示-03"> 
</center>
<p>可以看到<code>master</code>分支和<code>develop</code>分支指向了不同的提交记录，接下来我们将<code>develop</code>分支合并到<code>master</code>分支中</p> 
<pre data-index="11" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> merge develop
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859354.png" alt="03-图示-01"> 
</center>
<p>执行上面的指令后，产生了一个新的提交记录C4，由C4我们可以遍历之前所有的提交记录，但是此时<code>master</code>分支和<code>develop</code>分支仍然指向不同的提交记录。继续切换到<code>develop</code>分支，将<code>master</code>分支合并到<code>develop</code>分支中：</p> 
<pre data-index="12" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> checkout develop
<span class="token function">git</span> merge master
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li><li style="color: rgb(153, 153, 153);">2</li></ul></pre> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859349.png" alt="03-图示-01"> 
</center>
<h3><a name="t8"></a><a id="_162"></a>打标签</h3> 
<p>使用Git可以给指定提交打上标签，用来突出显示这个提交，比如将提交标记为<code>v1.0</code>、<code>v2.0</code>，等等</p> 
<ul><li><strong>列举标签</strong></li></ul> 
<p>使用如下命令即可列出所有标签</p> 
<pre data-index="13" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> tag
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<p>当标签太多时，可以使用如下指令列出包含指定字符的标签</p> 
<pre data-index="14" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> tag -l <span class="token string">"v1.*"</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<ul><li><strong>创建标签</strong></li></ul> 
<p>添加<code>-a</code>选项即可创建标签，如下：</p> 
<pre data-index="15" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> tag -a v1.0 -m <span class="token string">"version 1.0"</span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<p>如上命令即可为当前提交创建一个标签，标签名为<code>v1.0</code>，<code>-m</code>选项后就是该标签的附注信息</p> 
<ul><li><strong>推送标签</strong></li></ul> 
<p>只使用<code>git push</code>命令在默认情况下不会将标签推送到 远程仓库，在创建标签后需要执行如下命令将指定标签推送到远程仓库：</p> 
<pre data-index="16" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> push origin <span class="token operator"><</span>tagname<span class="token operator">></span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<p>如果要推送多个新标签，可以使用<code>git push</code>的<code>--tags</code>选项将所有标签推送到远程仓库：</p> 
<pre data-index="17" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> push origin --tags
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<ul><li><strong>删除标签</strong></li></ul> 
<p>使用<code>git tag</code>的<code>-d</code>选项即可删掉本地仓库上的指定标签，如下：</p> 
<pre data-index="18" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> tag -d <span class="token operator"><</span>tagname<span class="token operator">></span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<p>但是该指令不会删除远程仓库中的标签 ，还需要使用如下命令来更新远程仓库：</p> 
<pre data-index="19" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> push remote :refs/tags/<span class="token operator"><</span>tagname<span class="token operator">></span>
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<ul><li><strong>切换标签</strong></li></ul> 
<p>使用<code>git checkout <tagname></code>指令即可将git仓库的HEAD指针指向标签所在的提交，如下：</p> 
<pre data-index="20" class="prettyprint"><code class="prism language-bash has-numbering" onclick="mdcp.copyCode(event)" style="position: unset;"><span class="token function">git</span> checkout v1.0
<div class="hljs-button {2}" data-title="复制"></div></code><ul class="pre-numbering" style=""><li style="color: rgb(153, 153, 153);">1</li></ul></pre> 
<h2><a name="t9"></a><a id="VScodeGit_210"></a>VScode中的Git使用</h2> 
<h3><a name="t10"></a><a id="_212"></a>本地仓库操作</h3> 
<p>当对仓库已经被跟踪的文件进行修改的时候，会有三种文件状态。如图：</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859356.png" alt="01-vscode-图标说明-01"> 
</center>
<ul><li>M(Modify)，表示该文件存在修改</li><li>D(Delete)，表示该文件被删除</li><li>U(Update)，表示该文件是新添加的</li></ul> 
<p>选中文件即可查看已进行的修改</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859414.png" alt="01-vscode-图标说明-03"> 
</center>
<p>接下来可以对这些更改进行处理，可以选择放弃修改或者保存修改，选择放弃修改的话，该文件就会回退到上次保存的版本</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859419.png" alt="01-vscode-图标说明-02"> 
</center>
<p>也可以点击上面的图标对所有更改进行处理</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859413.png" alt="01-vscode-图标说明-04"> 
</center>
<p>我们选择保存所有修改，所有已修改文件就会保存到暂存区，对应的git命令为<code>git add .</code></p> 
<p>接下来将暂存区的改动提交到本地的版本库，点击上方的“√”，对应git命令<code>git commit</code>，然后添加message即可</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859415.png" alt="01-vscode-图标说明-05"> 
</center>
<p>这时候所有的修改就已经处理完毕了</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859401.png" alt="01-vscode-图标说明-06"> 
</center>
<h3><a name="t11"></a><a id="_244"></a>推送到远程仓库</h3> 
<p>将本地仓库上的修改推送到远程仓库，对应git命令<code>git push</code></p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106202847534.png" alt="01-vscode-图标说明-08"> 
</center>
<p>一般情况下，VScode会弹出账号密码的输入窗口进行登录</p> 
<p>最后查看远程仓库：</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106202819282.png" alt="01-vscode-图标说明-09"> 
</center>
<h3><a name="t12"></a><a id="_256"></a>从远程仓库拉取</h3> 
<p>与推送类似，如图，对应git命令<code>git pull</code></p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859437.png" alt="01-vscode-图标说明-11"> 
</center>
<h3><a name="t13"></a><a id="_262"></a>分支管理</h3> 
<p>VScode可以直接在左下角创建/切换分支</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106202746910.png" alt="01-vscode-图标说明-10"> 
</center>
<p>合并分支</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859473.png" alt="01-vscode-图标说明-12"> 
</center>
<p>如果待合并的分支上的修改和master没有冲突，就可以直接合并。但是在多人协作时常常会出现两个分支存在不同修改的情况，这时候就要对这些冲突进行处理：</p> 
<center> 
 <img src="https://img-blog.csdnimg.cn/20201106201859462.png" alt="01-vscode-图标说明-13"> 
</center>
<h3><a name="t14"></a><a id="GitLens_278"></a>GitLens插件</h3> 
<p>使用VScode自带的git支持对于个人开发来说已经足够了，但是在应对团队协作时的文件冲突时还略显不足，这时候我们可以借助VScode中的GitLens插件，使用方法详见<a href="https://www.jianshu.com/p/a91cb8a2e55d">git源代码管理插件GitLens</a></p> 
<h2><a name="t15"></a><a id="_284"></a>参考</h2> 
<p><a href="https://developer.ibm.com/zh/articles/j-lo-git-mange/">Git 分支管理最佳实践</a></p> 
<p><a href="https://www.liaoxuefeng.com/wiki/896043488029600">Git教程</a></p> 
<p><a href="https://blog.csdn.net/weixin_43314519/article/details/107572206">VScode 结合git的全面使用流程，再也不用记住git的命令了！</a></p> 
<p><a href="https://oschina.gitee.io/learn-git-branching/">https://oschina.gitee.io/learn-git-branching/</a></p>
                </div><div><div></div></div>
                <link href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/editerView/markdown_views-22a2fefd3b.css" rel="stylesheet">
                <link href="https://csdnimg.cn/release/blogv2/dist/mdeditor/css/style-4f8fbf9108.css" rel="stylesheet">
        </div>
        <div id="treeSkill" style="display: block;"><div class="skill-tree-box"><div class="skill-tree-head">文章知识点与官方知识档案匹配，可进一步学习相关知识</div><div class="skill-tree-body"><div class="skill-tree-item"><span class="skill-tree-href"><a data-report-click="{"spm":"1001.2101.3001.6866","dest":"https://edu.csdn.net/skill/gml/gml-62c30f9c31f64a1d96af732c47c93f04"}" href="https://edu.csdn.net/skill/gml/gml-62c30f9c31f64a1d96af732c47c93f04" target="_blank">CS入门技能树</a><i></i><a data-report-click="{"spm":"1001.2101.3001.6866","dest":"https://edu.csdn.net/skill/gml/gml-62c30f9c31f64a1d96af732c47c93f04"}" href="https://edu.csdn.net/skill/gml/gml-62c30f9c31f64a1d96af732c47c93f04" target="_blank">Git入门</a><i></i><a data-report-click="{"spm":"1001.2101.3001.6866","dest":"https://edu.csdn.net/skill/gml/gml-62c30f9c31f64a1d96af732c47c93f04"}" href="https://edu.csdn.net/skill/gml/gml-62c30f9c31f64a1d96af732c47c93f04" target="_blank">Git简介</a></span><span class="skill-tree-con"><span class="skill-tree-count">13813</span> 人正在系统学习中</span></div></div></div></div>
    </article>
