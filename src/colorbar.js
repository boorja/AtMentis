import * as d3 from 'd3';

export function drawColorBar(containerId, maxDegree, options = {}) {
  const {
    width = 300,
    height = 50,
    gradientColors = d3.interpolateRdYlBu,
    minLabel = 'Min Degree',
    maxLabel = 'Max Degree'
  } = options;

  const container = d3.select(containerId);
  container.selectAll('*').remove();

  const svg = container
    .append('svg')
    .attr('width', width)
    .attr('height', height + 30);

  const colorScale = d3.scaleSequential(gradientColors)
    .domain([Math.log(maxDegree + 1), Math.log(1)]);

  const defs = svg.append('defs');
  const linearGradient = defs.append('linearGradient')
    .attr('id', 'color-gradient');

  linearGradient.append('stop')
    .attr('offset', '0%')
    .attr('stop-color', gradientColors(1)); 
  
  linearGradient.append('stop')
    .attr('offset', '100%')
    .attr('stop-color', gradientColors(0));

  svg.append('rect')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', width)
    .attr('height', height)
    .style('fill', 'url(#color-gradient)');

  svg.append('text')
    .attr('x', 0)
    .attr('y', height + 20)
    .attr('fill', 'black')
    .attr('font-size', '12px')
    .text(maxLabel);

  svg.append('text')
    .attr('x', width - 60)
    .attr('y', height + 20)
    .attr('fill', 'black')
    .attr('font-size', '12px')
    .text(minLabel);
}
