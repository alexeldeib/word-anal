"""Interactive visualization generation using D3.js."""

import json
from typing import Dict, List
import pandas as pd


class VisualizationGenerator:
    """Generate interactive D3.js visualizations for word permutation analysis."""

    def __init__(self, data_processor):
        """
        Initialize the visualization generator.

        Args:
            data_processor: DataProcessor instance with analysis results
        """
        self.data_processor = data_processor

    def generate_html(
        self,
        output_path: str = "word_analysis_visualization.html",
        title: str = "Word Permutation Analysis"
    ):
        """
        Generate a complete HTML file with D3.js visualizations.

        Args:
            output_path: Path to save HTML file
            title: Title for the visualization page
        """
        viz_data = self.data_processor.prepare_visualization_data(bins=50)

        html_content = self._create_html_template(title, viz_data)

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"Visualization saved to {output_path}")

    def _create_html_template(self, title: str, viz_data: Dict) -> str:
        """Create the complete HTML template with embedded D3.js visualizations."""
        data_json = json.dumps(viz_data, indent=2)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }}

        .viz-section {{
            margin-bottom: 50px;
        }}

        .viz-section h2 {{
            color: #444;
            margin-bottom: 20px;
            border-bottom: 2px solid #4a90e2;
            padding-bottom: 10px;
        }}

        .chart-container {{
            margin: 20px 0;
        }}

        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        .stats-table th,
        .stats-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        .stats-table th {{
            background-color: #4a90e2;
            color: white;
            font-weight: 600;
        }}

        .stats-table tr:hover {{
            background-color: #f5f5f5;
        }}

        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            font-size: 12px;
            opacity: 0;
            transition: opacity 0.2s;
        }}

        .legend {{
            display: flex;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 0 15px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border-radius: 3px;
        }}

        .axis text {{
            font-size: 12px;
        }}

        .axis path,
        .axis line {{
            stroke: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">Interactive analysis of valid English words from letter permutations</p>

        <!-- Statistics Table -->
        <div class="viz-section">
            <h2>Summary Statistics</h2>
            <div id="stats-table"></div>
        </div>

        <!-- Distribution Comparison -->
        <div class="viz-section">
            <h2>Distribution Comparison</h2>
            <div class="legend" id="legend"></div>
            <div class="chart-container" id="overlay-histogram"></div>
        </div>

        <!-- Individual Distributions -->
        <div class="viz-section">
            <h2>Individual Distributions</h2>
            <div class="chart-container" id="individual-histograms"></div>
        </div>

        <!-- Box Plot Comparison -->
        <div class="viz-section">
            <h2>Box Plot Comparison</h2>
            <div class="chart-container" id="box-plot"></div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        // Data
        const data = {data_json};

        // Color scheme
        const colorScale = d3.scaleOrdinal()
            .domain([5, 6, 7])
            .range(['#4a90e2', '#e74c3c', '#2ecc71']);

        // Create statistics table
        function createStatsTable() {{
            const container = d3.select('#stats-table');

            const table = container.append('table')
                .attr('class', 'stats-table');

            const thead = table.append('thead');
            const tbody = table.append('tbody');

            // Headers
            const headers = ['Word Length', 'Permutations', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'];
            thead.append('tr')
                .selectAll('th')
                .data(headers)
                .enter()
                .append('th')
                .text(d => d);

            // Rows
            const rows = tbody.selectAll('tr')
                .data(data.comparison_stats)
                .enter()
                .append('tr');

            rows.selectAll('td')
                .data(d => [
                    d.word_length,
                    d.total_permutations.toLocaleString(),
                    d.mean.toFixed(2),
                    d.median.toFixed(2),
                    d.std.toFixed(2),
                    d.min,
                    d.max,
                    d.q25.toFixed(2),
                    d.q75.toFixed(2)
                ])
                .enter()
                .append('td')
                .text(d => d);
        }}

        // Create legend
        function createLegend() {{
            const legend = d3.select('#legend');

            Object.keys(data.distributions).forEach(wordLength => {{
                const item = legend.append('div')
                    .attr('class', 'legend-item');

                item.append('div')
                    .attr('class', 'legend-color')
                    .style('background-color', colorScale(wordLength));

                item.append('span')
                    .text(`${{wordLength}}-letter words`);
            }});
        }}

        // Create overlay histogram
        function createOverlayHistogram() {{
            const margin = {{top: 20, right: 30, bottom: 50, left: 60}};
            const width = 1000 - margin.left - margin.right;
            const height = 400 - margin.top - margin.bottom;

            const svg = d3.select('#overlay-histogram')
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

            // Get all histogram data
            const allData = [];
            Object.entries(data.distributions).forEach(([wordLength, dist]) => {{
                dist.histogram.forEach(d => {{
                    allData.push({{
                        ...d,
                        wordLength: +wordLength
                    }});
                }});
            }});

            // Scales
            const x = d3.scaleLinear()
                .domain([0, d3.max(allData, d => d.bin_end)])
                .range([0, width]);

            const y = d3.scaleLinear()
                .domain([0, d3.max(allData, d => d.frequency)])
                .range([height, 0]);

            // X axis
            svg.append('g')
                .attr('class', 'axis')
                .attr('transform', `translate(0,${{height}})`)
                .call(d3.axisBottom(x))
                .append('text')
                .attr('x', width / 2)
                .attr('y', 40)
                .attr('fill', 'black')
                .style('text-anchor', 'middle')
                .text('Number of Valid Words');

            // Y axis
            svg.append('g')
                .attr('class', 'axis')
                .call(d3.axisLeft(y).tickFormat(d3.format('.1%')))
                .append('text')
                .attr('transform', 'rotate(-90)')
                .attr('y', -50)
                .attr('x', -height / 2)
                .attr('fill', 'black')
                .style('text-anchor', 'middle')
                .text('Frequency');

            // Draw histograms for each word length
            Object.entries(data.distributions).forEach(([wordLength, dist]) => {{
                svg.selectAll(`.bar-${{wordLength}}`)
                    .data(dist.histogram)
                    .enter()
                    .append('rect')
                    .attr('class', `bar-${{wordLength}}`)
                    .attr('x', d => x(d.bin_start))
                    .attr('width', d => Math.max(0, x(d.bin_end) - x(d.bin_start) - 1))
                    .attr('y', d => y(d.frequency))
                    .attr('height', d => height - y(d.frequency))
                    .attr('fill', colorScale(wordLength))
                    .attr('opacity', 0.6)
                    .on('mouseover', function(event, d) {{
                        d3.select('#tooltip')
                            .style('opacity', 1)
                            .html(`
                                <strong>${{wordLength}}-letter words</strong><br/>
                                Range: ${{d.bin_start.toFixed(1)}} - ${{d.bin_end.toFixed(1)}}<br/>
                                Count: ${{d.count}}<br/>
                                Frequency: ${{(d.frequency * 100).toFixed(2)}}%
                            `)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 10) + 'px');
                    }})
                    .on('mouseout', function() {{
                        d3.select('#tooltip').style('opacity', 0);
                    }});
            }});
        }}

        // Create individual histograms
        function createIndividualHistograms() {{
            const container = d3.select('#individual-histograms');
            const margin = {{top: 20, right: 30, bottom: 50, left: 60}};
            const width = 450 - margin.left - margin.right;
            const height = 300 - margin.top - margin.bottom;

            Object.entries(data.distributions).forEach(([wordLength, dist]) => {{
                const chartDiv = container.append('div')
                    .style('display', 'inline-block')
                    .style('margin', '10px');

                chartDiv.append('h3')
                    .text(`${{wordLength}}-letter Words`)
                    .style('text-align', 'center')
                    .style('color', colorScale(wordLength));

                const svg = chartDiv.append('svg')
                    .attr('width', width + margin.left + margin.right)
                    .attr('height', height + margin.top + margin.bottom)
                    .append('g')
                    .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

                const x = d3.scaleLinear()
                    .domain([0, d3.max(dist.histogram, d => d.bin_end)])
                    .range([0, width]);

                const y = d3.scaleLinear()
                    .domain([0, d3.max(dist.histogram, d => d.count)])
                    .range([height, 0]);

                svg.append('g')
                    .attr('class', 'axis')
                    .attr('transform', `translate(0,${{height}})`)
                    .call(d3.axisBottom(x).ticks(5));

                svg.append('g')
                    .attr('class', 'axis')
                    .call(d3.axisLeft(y).ticks(5));

                svg.selectAll('.bar')
                    .data(dist.histogram)
                    .enter()
                    .append('rect')
                    .attr('x', d => x(d.bin_start))
                    .attr('width', d => Math.max(0, x(d.bin_end) - x(d.bin_start) - 1))
                    .attr('y', d => y(d.count))
                    .attr('height', d => height - y(d.count))
                    .attr('fill', colorScale(wordLength));
            }});
        }}

        // Create box plot
        function createBoxPlot() {{
            const margin = {{top: 20, right: 30, bottom: 50, left: 60}};
            const width = 600 - margin.left - margin.right;
            const height = 400 - margin.top - margin.bottom;

            const svg = d3.select('#box-plot')
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

            const wordLengths = Object.keys(data.distributions).map(Number);

            const x = d3.scaleBand()
                .domain(wordLengths)
                .range([0, width])
                .padding(0.3);

            const allValues = [];
            Object.values(data.distributions).forEach(dist => {{
                allValues.push(...dist.raw_counts);
            }});

            const y = d3.scaleLinear()
                .domain([0, d3.max(allValues)])
                .range([height, 0]);

            svg.append('g')
                .attr('class', 'axis')
                .attr('transform', `translate(0,${{height}})`)
                .call(d3.axisBottom(x))
                .append('text')
                .attr('x', width / 2)
                .attr('y', 40)
                .attr('fill', 'black')
                .style('text-anchor', 'middle')
                .text('Word Length');

            svg.append('g')
                .attr('class', 'axis')
                .call(d3.axisLeft(y))
                .append('text')
                .attr('transform', 'rotate(-90)')
                .attr('y', -50)
                .attr('x', -height / 2)
                .attr('fill', 'black')
                .style('text-anchor', 'middle')
                .text('Valid Word Count');

            wordLengths.forEach(wordLength => {{
                const stats = data.comparison_stats.find(d => d.word_length === wordLength);
                const xPos = x(wordLength) + x.bandwidth() / 2;

                // Interquartile range box
                svg.append('rect')
                    .attr('x', x(wordLength))
                    .attr('y', y(stats.q75))
                    .attr('width', x.bandwidth())
                    .attr('height', y(stats.q25) - y(stats.q75))
                    .attr('fill', colorScale(wordLength))
                    .attr('opacity', 0.7);

                // Median line
                svg.append('line')
                    .attr('x1', x(wordLength))
                    .attr('x2', x(wordLength) + x.bandwidth())
                    .attr('y1', y(stats.median))
                    .attr('y2', y(stats.median))
                    .attr('stroke', 'black')
                    .attr('stroke-width', 2);

                // Whiskers
                svg.append('line')
                    .attr('x1', xPos)
                    .attr('x2', xPos)
                    .attr('y1', y(stats.min))
                    .attr('y2', y(stats.q25))
                    .attr('stroke', 'black');

                svg.append('line')
                    .attr('x1', xPos)
                    .attr('x2', xPos)
                    .attr('y1', y(stats.q75))
                    .attr('y2', y(stats.max))
                    .attr('stroke', 'black');

                // Mean marker
                svg.append('circle')
                    .attr('cx', xPos)
                    .attr('cy', y(stats.mean))
                    .attr('r', 4)
                    .attr('fill', 'white')
                    .attr('stroke', 'black')
                    .attr('stroke-width', 2);
            }});
        }}

        // Initialize all visualizations
        createStatsTable();
        createLegend();
        createOverlayHistogram();
        createIndividualHistograms();
        createBoxPlot();
    </script>
</body>
</html>
"""

    def generate_notebook_display(self):
        """
        Generate visualizations for display in Jupyter notebook.

        Returns:
            HTML object for notebook display
        """
        from IPython.display import HTML

        viz_data = self.data_processor.prepare_visualization_data(bins=50)
        html_content = self._create_html_template(
            "Word Permutation Analysis",
            viz_data
        )

        return HTML(html_content)
