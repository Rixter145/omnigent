import { describe, expect, it } from "vitest";
import { isTempConvId, newTempConvId, TEMP_CONV_ID_PREFIX } from "./tempConversationId";

describe("tempConversationId", () => {
  it("isTempConvId recognizes temp ids and rejects real ones", () => {
    expect(isTempConvId("temp:0a1b2c3d")).toBe(true);
    expect(isTempConvId(TEMP_CONV_ID_PREFIX)).toBe(true);
    expect(isTempConvId("conv_abc123")).toBe(false);
    expect(isTempConvId("pend_conv_1")).toBe(false);
    expect(isTempConvId(null)).toBe(false);
    expect(isTempConvId(undefined)).toBe(false);
  });

  it("newTempConvId mints a temp: id with 8 hex chars, and isTempConvId accepts it", () => {
    for (let i = 0; i < 50; i++) {
      const id = newTempConvId();
      expect(id).toMatch(/^temp:[0-9a-f]{8}$/);
      expect(isTempConvId(id)).toBe(true);
    }
  });
});
