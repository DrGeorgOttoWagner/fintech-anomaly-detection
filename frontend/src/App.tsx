import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Spinner } from './components/ui'
import { SessionProvider, useSession } from './lib/session'
import AnomalyExplorer from './pages/AnomalyExplorer'
import Benchmark from './pages/Benchmark'
import Datasets from './pages/Datasets'
import Evaluation from './pages/Evaluation'
import Explore from './pages/Explore'
import Inference from './pages/Inference'
import Overview from './pages/Overview'
import Reports from './pages/Reports'
import Schema from './pages/Schema'
import Studio from './pages/Studio'

function Shell() {
  const { ready } = useSession()
  if (!ready) return <div className="p-10"><Spinner label="Connecting to the local FTAD backend…" /></div>
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/schema" element={<Schema />} />
        <Route path="/explore" element={<Explore />} />
        <Route path="/studio" element={<Studio />} />
        <Route path="/benchmark" element={<Benchmark />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/anomalies" element={<AnomalyExplorer />} />
        <Route path="/inference" element={<Inference />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="*" element={<Overview />} />
      </Routes>
    </Layout>
  )
}

export default function App() {
  return (
    <SessionProvider>
      <Shell />
    </SessionProvider>
  )
}
