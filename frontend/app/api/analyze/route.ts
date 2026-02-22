const FLASK_URL = process.env.FLASK_URL || "http://localhost:8000"

export async function POST(req: Request) {
  const incomingForm = await req.formData()

  try {
    const response = await fetch(`${FLASK_URL}/analyze`, {
      method: "POST",
      body: incomingForm,
    })

    const data = await response.json()

    if (response.ok) {
      return Response.json({ result: data })
    }

    return Response.json(
      { error: data.error ?? `Backend error (${response.status})` },
      { status: response.status }
    )
  } catch (err) {
    console.error("Flask backend unreachable:", err)
    return Response.json({ error: "Analysis service is unavailable." }, { status: 503 })
  }
}
