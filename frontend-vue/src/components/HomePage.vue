<template>
    <div class="homepage">
      <section class="hero">
        <h1 class="hero-title">GenHack4</h1>
        <p class="hero-subtitle">
          Explore temperature biases between ERA5 reanalysis and local station observations,
          understand UHI patterns, and run advanced AI/ML models — all in one dashboard.
        </p>
  
        <button class="btn btn-primary hero-btn" @click="$emit('start')">
          Select a City to Begin →
        </button>
      </section>
  
      <section class="info-section">
        <div class="card">
          <h2 class="card-title">What This Platform Does</h2>
          <p class="card-text">
            This tool helps you evaluate the accuracy and behavior of ERA5 temperature products
            when compared to local weather station data. It provides:
          </p>
  
          <ul class="feature-list">
            <li>ERA5 vs Station temperature comparison</li>
            <li>Heatwave and extreme event detection</li>
            <li>Urban Heat Island (UHI) diagnostics</li>
            <li>Machine Learning bias correction models</li>
            <li>Interactive time-series and scatter plots</li>
          </ul>
        </div>
  
        <div class="card">
          <h2 class="card-title">Why ERA5 vs Station Data?</h2>
          <p class="card-text">
            ERA5 is a global reanalysis dataset, providing consistent long-term climate data.
            However, it can underestimate local temperature extremes, especially in dense urban areas.
          </p>
  
          <p class="card-highlight">
            Our platform quantifies this bias and shows <strong>when, where, and why</strong>
            ERA5 deviates from ground truth.
          </p>
        </div>
      </section>
    </div>
  </template>
  
  <script>
  export default {
    name: "HomePage",
    emits: ["start"],
  };
  </script>
  
  <style scoped>
  .homepage {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem;
    color: var(--color-text);
  }
  
  /* ===== HERO SECTION ===== */
  .hero {
    text-align: center;
    padding: 3rem 1rem;
    margin-bottom: 2rem;
    background: var(--color-bg-secondary);
    border-radius: 10px;
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow-card);
  }
  
  .hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
  }
  
  .hero-subtitle {
    font-size: 1.125rem;
    max-width: 700px;
    margin: 0 auto 2rem auto;
    color: var(--color-text-secondary);
  }
  
  .hero-btn {
    padding: 0.8rem 1.5rem;
    font-size: 1rem;
  }
  
  /* ===== INFO CARDS ===== */
  .info-section {
    display: grid;
    gap: 1.5rem;
    grid-template-columns: 1fr 1fr;
  }
  
  .card {
    background: var(--color-bg-secondary);
    border: 1px solid var(--color-border);
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: var(--shadow-card);
  }
  
  .card-title {
    font-size: 1.3rem;
    margin-bottom: 1rem;
  }
  
  .card-text {
    font-size: 0.95rem;
    margin-bottom: 1rem;
    color: var(--color-text-secondary);
  }
  
  .card-highlight {
    background: var(--color-bg-tertiary);
    padding: 0.75rem;
    border-radius: 6px;
    border-left: 4px solid var(--color-primary);
  }
  
  .feature-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  
  .feature-list li {
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
    padding-left: 0.5rem;
  }
  
  @media (max-width: 900px) {
    .info-section {
      grid-template-columns: 1fr;
    }
  }
  </style>
  