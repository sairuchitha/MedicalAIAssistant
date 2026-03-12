export default function CitationList({ citations }) {
  if (!citations?.length) return null;

  return (
    <div className="citation-list">
      <h4>Citations</h4>
      <ol>
        {citations.map((c) => (
          <li key={c.id}>
            [{c.id}] {c.date} | {c.note_type} | {c.section_name} | Note {c.note_id}
          </li>
        ))}
      </ol>
    </div>
  );
}
