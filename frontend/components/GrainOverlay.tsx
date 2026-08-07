/** Film grain + a vignette, done in CSS/SVG rather than a WebGL postprocessing pass - zero
 * extra GPU cost on top of the scene, works fine on mobile, and needs no
 * @react-three/postprocessing dependency (not installed in this project) for what is
 * otherwise two purely cosmetic effects. A plain absolutely-positioned overlay ABOVE the
 * canvas, not inside it - this is exactly the "2026 cohort" move described in the redesign
 * research (Codrops et al favor CSS noise/grain over heavy WebGL for atmosphere). */

const NOISE_SVG =
  '<svg xmlns="http://www.w3.org/2000/svg">' +
  '<filter id="n"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" stitchTiles="stitch"/></filter>' +
  '<rect width="100%" height="100%" filter="url(#n)"/>' +
  "</svg>";
// encodeURIComponent (not hand-escaped %23/%25) so the quotes/# inside the SVG never have to
// be manually percent-escaped - one less place to get a data: URI subtly wrong.
const NOISE_DATA_URL = `data:image/svg+xml,${encodeURIComponent(NOISE_SVG)}`;

export function GrainOverlay() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-md">
      <div
        className="absolute inset-0"
        style={{ backgroundImage: `url("${NOISE_DATA_URL}")`, opacity: 0.05, mixBlendMode: "overlay" }}
      />
      <div
        className="absolute inset-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,0.35) 100%)" }}
      />
    </div>
  );
}
