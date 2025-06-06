<div id="top" class="">

<div align="center" class="text-center">
<h1>SLM</h1>
<p><em>Transforming dialogues into intelligent conversations effortlessly.</em></p>

<img alt="last-commit" src="https://img.shields.io/github/last-commit/Md-nexus/SLM?style=flat&amp;logo=git&amp;logoColor=white&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<img alt="repo-top-language" src="https://img.shields.io/github/languages/top/Md-nexus/SLM?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<img alt="repo-language-count" src="https://img.shields.io/github/languages/count/Md-nexus/SLM?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<p><em>Built with the tools and technologies:</em></p>
<img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&amp;logo=Python&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">
</div>
<br>
<hr>
<h2>Table of Contents</h2>
<ul class="list-disc pl-4 my-0">
<li class="my-0"><a href="#overview">Overview</a></li>
<li class="my-0"><a href="#getting-started">Getting Started</a>
<ul class="list-disc pl-4 my-0">
<li class="my-0"><a href="#prerequisites">Prerequisites</a></li>
<li class="my-0"><a href="#installation">Installation</a></li>
<li class="my-0"><a href="#usage">Usage</a></li>
<li class="my-0"><a href="#testing">Testing</a></li>
</ul>
</li>
</ul>
<hr>
<h2>Overview</h2>
<p>SLM is a powerful developer tool designed to streamline the process of building conversational AI applications.</p>
<p><strong>Why SLM?</strong></p>
<p>This project aims to simplify the journey from raw conversational data to robust language models. The core features include:</p>
<ul class="list-disc pl-4 my-0">
<li class="my-0">🎯 <strong>Data Formatting:</strong> Extracts and organizes conversational data into structured text for easy usability.</li>
<li class="my-0">🚀 <strong>Model Training:</strong> Facilitates efficient training of sequence language models using PyTorch, optimizing performance.</li>
<li class="my-0">📥 <strong>Data Loading:</strong> Seamlessly loads the "daily_dialog" dataset, providing rich conversational examples for model training.</li>
<li class="my-0">🔤 <strong>Tokenization:</strong> Processes text data into numerical formats, essential for machine learning tasks.</li>
<li class="my-0">🧠 <strong>Model Architecture:</strong> Implements a GRU-based sequence learning model, capturing sequential dependencies for NLP tasks.</li>
<li class="my-0">🔍 <strong>Data Integrity Checks:</strong> Identifies non-printable characters in datasets, ensuring high data quality.</li>
</ul>
<hr>
<h2>Getting Started</h2>
<h3>Prerequisites</h3>
<p>This project requires the following dependencies:</p>
<ul class="list-disc pl-4 my-0">
<li class="my-0"><strong>Programming Language:</strong> Python</li>
<li class="my-0"><strong>Package Manager:</strong> Conda</li>
</ul>
<h3>Installation</h3>
<p>Build SLM from the source and intsall dependencies:</p>
<ol>
<li class="my-0">
<p><strong>Clone the repository:</strong></p>
<pre><code class="language-sh">❯ git clone https://github.com/Md-nexus/SLM
</code></pre>
</li>
<li class="my-0">
<p><strong>Navigate to the project directory:</strong></p>
<pre><code class="language-sh">❯ cd SLM
</code></pre>
</li>
<li class="my-0">
<p><strong>Install the dependencies:</strong></p>
</li>
</ol>
<p><strong>Using <a href="https://docs.conda.io/">conda</a>:</strong></p>
<pre><code class="language-sh">❯ conda env create -f conda.yml
</code></pre>
<h3>Usage</h3>
<p>Run the project with:</p>
<p><strong>Using <a href="https://docs.conda.io/">conda</a>:</strong></p>
<pre><code class="language-sh">conda activate {venv}
python {entrypoint}
</code></pre>
<h3>Testing</h3>
<p>Slm uses the {<strong>test_framework</strong>} test framework. Run the test suite with:</p>
<p><strong>Using <a href="https://docs.conda.io/">conda</a>:</strong></p>
<pre><code class="language-sh">conda activate {venv}
pytest
</code></pre>
<hr>
<div align="left" class=""><a href="#top">⬆ Return</a></div>
<hr></div>
