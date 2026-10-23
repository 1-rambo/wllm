import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

const e2eConfigUrl = new URL("../../evaluation/config/e2e.json", import.meta.url);
const preregistrationUrl = new URL(
  "../../evaluation/config/e2e-preregistration.json",
  import.meta.url
);

describe("rapid E2E configuration", () => {
  it("uses one smoke entry, one formal repeat, and the reduced pressure ladder", async () => {
    const config = JSON.parse(await readFile(e2eConfigUrl, "utf8"));

    expect(config.layout).toEqual({
      appTabs: 6,
      artifexTabs: 3,
      pressureTabs: 1,
      totalTabs: 7
    });
    expect(config.pressure.smokeSchedule).toEqual([
      { id: "clear-1", triangles: 0 }
    ]);
    expect(
      config.pressure.schedule.map(
        (entry: { triangles: number }) => entry.triangles
      )
    ).toEqual([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 0]);
    expect(config.modes.pilot).toBeUndefined();
    expect(config.modes.smoke).toMatchObject({
      appTabs: 2,
      repeats: 1,
      judge: false
    });
    expect(config.modes.formal).toMatchObject({
      appTabs: 6,
      artifexTabs: 3,
      pressureTabs: 1,
      totalTabs: 7,
      repeats: 1,
      judge: false
    });
    expect(config.priority.sequence).toEqual([0, 3, 1, 4, 2, 5]);
  });

  it("freezes the rapid protocol without acceptance gates", async () => {
    const configText = await readFile(e2eConfigUrl, "utf8");
    const preregistration = JSON.parse(
      await readFile(preregistrationUrl, "utf8")
    );

    expect(preregistration.configSha256).toBe(
      createHash("sha256").update(configText).digest("hex")
    );
    expect(preregistration.protocol).toMatchObject({
      repeats: 1,
      acceptanceGates: false
    });
    expect(preregistration.gates).toBeUndefined();
  });
});
