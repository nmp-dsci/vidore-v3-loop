/** A run's configuration, the same five sections in the same order for every run (eval/setup.py). */
import { Link } from 'react-router-dom';
import type { SetupSection } from './api';

export function SetupTable({ setup }: { setup: SetupSection[] | undefined }) {
  if (!setup) return null;
  return (
    <>
      <h2>Setup</h2>
      <div className="tw">
        <table className="fit setup">
          <tbody>
            {setup.flatMap((sec) =>
              sec.rows.map((r, i) => (
                <tr key={`${sec.section}-${r.k}`} className={i === 0 ? 'secstart' : ''}>
                  {i === 0 && (
                    <th scope="rowgroup" rowSpan={sec.rows.length}>
                      {sec.section}
                    </th>
                  )}
                  <td className="muted nw">{r.k}</td>
                  <td className={r.mono ? 'mono small' : ''}>
                    {r.link ? (
                      r.link.startsWith('/') ? (
                        <Link to={r.link}>{r.v}</Link>
                      ) : (
                        <a href={r.link} target="_blank" rel="noreferrer">
                          {r.v}
                        </a>
                      )
                    ) : (
                      r.v
                    )}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
