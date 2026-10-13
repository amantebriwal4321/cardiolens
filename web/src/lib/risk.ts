export function riskRgb(probability: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0, [47, 157, 132]],
    [0.5, [230, 177, 90]],
    [1, [217, 78, 72]],
  ];
  const clamped = Math.min(1, Math.max(0, probability));
  let index = 0;
  while (index < stops.length - 2 && clamped > stops[index + 1][0]) index += 1;
  const [start, from] = stops[index];
  const [end, to] = stops[index + 1];
  const mix = (clamped - start) / (end - start);
  return from.map((channel, channelIndex) =>
    Math.round(channel + (to[channelIndex] - channel) * mix),
  ) as [number, number, number];
}

export function riskCss(probability: number): string {
  const [red, green, blue] = riskRgb(probability);
  return `rgb(${red} ${green} ${blue})`;
}

export function riskHex(probability: number): string {
  const [red, green, blue] = riskRgb(probability);
  return `#${[red, green, blue].map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

export function likelihoodBand(probability: number): string {
  if (probability < 0.35) return "Lower estimated likelihood";
  if (probability < 0.65) return "Intermediate estimated likelihood";
  return "Higher estimated likelihood";
}

export function formatPercent(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}

export function formatMetric(value: number): string {
  return value.toFixed(3);
}
