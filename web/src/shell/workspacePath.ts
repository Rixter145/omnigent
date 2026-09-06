/** True for Windows drive-letter paths such as `C:/Users/me` or `C:\\Users\\me`. */
export function isWindowsDrivePath(path: string): boolean {
  return /^[A-Za-z]:[\\/]/.test(path);
}

/**
 * True when the path is already an absolute host path: POSIX `/...`
 * or a Windows drive path (`C:\\...` / `C:/...`).
 */
export function isHostAbsolutePath(path: string): boolean {
  return path.startsWith("/") || isWindowsDrivePath(path);
}
