import toLower from 'lodash/toLower';

export function getCryptoIcon(asset: string): string {
  return `https://www.cryptofonts.com/img/icons/${toLower(asset)}.svg`;
}
